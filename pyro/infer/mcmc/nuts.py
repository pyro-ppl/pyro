from __future__ import absolute_import, division, print_function

from collections import namedtuple

import torch

import pyro
import pyro.distributions as dist
from pyro.ops.integrator import single_step_velocity_verlet
from pyro.util import torch_isnan

from pyro.infer.mcmc.hmc import HMC

# sum_accept_probs and num_proposals are used to calculate
# the statistic accept_prob for Dual Averaging scheme;
# z_left_grads and z_right_grads are kept to avoid recalculating
# grads at left and right leaves
_TreeInfo = namedtuple("TreeInfo", ["z_left", "r_left", "z_left_grads",
                                    "z_right", "r_right", "z_right_grads",
                                    "z_proposal", "size", "turning", "diverging",
                                    "sum_accept_probs", "num_proposals"])


class NUTS(HMC):
    """
    No-U-Turn Sampler kernel, which provides an efficient and convenient way
    to run Hamiltonian Monte Carlo. The number of steps taken by the
    integrator is dynamically adjusted on each call to ``sample`` to ensure
    an optimal length for the Hamiltonian trajectory [1]. As such, the samples
    generated will typically have lower autocorrelation than those generated
    by the :class:`~pyro.infer.mcmc.HMC` kernel. Optionally, the NUTS kernel
    also provides the ability to adapt step size during the warmup phase.

    Refer to the `baseball example <https://github.com/uber/pyro/blob/dev/examples/baseball.py>`_
    to see how to do Bayesian inference in Pyro using NUTS.

    **References**

    [1] `The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo`,
    Matthew D. Hoffman, and Andrew Gelman

    :param model: Python callable containing Pyro primitives.
    :param float step_size: Determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics. If not specified, it will be set to 1.
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.

    Example:

        >>> true_coefs = torch.tensor([1., 2., 3.])
        >>> data = torch.randn(2000, 3)
        >>> dim = 3
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()
        >>>
        >>> def model(data):
        ...     coefs_mean = torch.zeros(dim)
        ...     coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
        ...     y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        ...     return y
        >>>
        >>> nuts_kernel = NUTS(model, adapt_step_size=True)
        >>> mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=300).run(data)
        >>> posterior = EmpiricalMarginal(mcmc_run, 'beta')
        >>> posterior.mean  # doctest: +SKIP
        tensor([ 0.9221,  1.9464,  2.9228])
    """

    def __init__(self, model, step_size=None, adapt_step_size=False, transforms=None):
        super(NUTS, self).__init__(model, step_size, adapt_step_size=adapt_step_size,
                                   transforms=transforms)

        self._max_tree_depth = 10  # from Stan
        # There are three conditions to stop doubling process:
        #     + Tree is becoming too big.
        #     + The trajectory is making a U-turn.
        #     + The probability of the states becoming negligible: p(z, r) << u,
        # here u is the "slice" variable introduced at the `self.sample(...)` method.
        # Denote E_p = -log p(z, r), E_u = -log u, the third condition is equivalent to
        #     sliced_energy := E_p - E_u >= some constant =: max_sliced_energy.
        # This also suggests the notion "diverging" in the implemenation:
        #     when the energy E_p diverges from E_u too much, we stop doubling.
        # Here, as suggested in [1], we set dE_max = 1000.
        self._max_sliced_energy = 1000

    def _is_turning(self, z_left, r_left, z_right, r_right):
        diff_left = 0
        diff_right = 0
        for name in self._r_dist:
            dz = z_right[name] - z_left[name]
            diff_left += (dz * r_left[name]).sum()
            diff_right += (dz * r_right[name]).sum()
        return diff_left < 0 or diff_right < 0

    def _build_basetree(self, z, r, z_grads, log_slice, direction, energy_current):
        step_size = self.step_size if direction == 1 else -self.step_size
        z_new, r_new, z_grads, potential_energy = single_step_velocity_verlet(
            z, r, self._potential_energy, step_size, z_grads=z_grads)
        energy_new = potential_energy + self._kinetic_energy(r_new)
        sliced_energy = energy_new + log_slice

        # As a part of the slice sampling process (see below), along the trajectory
        #     we eliminate states which p(z, r) < u, or dE > 0.
        # Due to this elimination (and stop doubling conditions),
        #     the size of binary tree might not equal to 2^tree_depth.
        tree_size = 1 if sliced_energy <= 0 else 0
        # Special case: Set diverging to True and accept prob to 0 if the
        # diverging trajectory returns `NaN` energy (e.g. in the case of
        # evaluating log prob of a value simulated using a large step size
        # for a constrained sample site).
        if torch_isnan(energy_new):
            diverging = True
            accept_prob = energy_new.new_tensor(0.0)
        else:
            diverging = (sliced_energy >= self._max_sliced_energy)
            delta_energy = energy_new - energy_current
            accept_prob = (-delta_energy).exp().clamp(max=1)
        return _TreeInfo(z_new, r_new, z_grads, z_new, r_new, z_grads,
                         z_new, tree_size, False, diverging, accept_prob, 1)

    def _build_tree(self, z, r, z_grads, log_slice, direction, tree_depth, energy_current):
        if tree_depth == 0:
            return self._build_basetree(z, r, z_grads, log_slice, direction, energy_current)

        # build the first half of tree
        half_tree = self._build_tree(z, r, z_grads, log_slice,
                                     direction, tree_depth-1, energy_current)
        z_proposal = half_tree.z_proposal

        # Check conditions to stop doubling. If we meet that condition,
        #     there is no need to build the other tree.
        if half_tree.turning or half_tree.diverging:
            return half_tree

        # Else, build remaining half of tree.
        # If we are going to the right, start from the right leaf of the first half.
        if direction == 1:
            z = half_tree.z_right
            r = half_tree.r_right
            z_grads = half_tree.z_right_grads
        else:  # otherwise, start from the left leaf of the first half
            z = half_tree.z_left
            r = half_tree.r_left
            z_grads = half_tree.z_left_grads
        other_half_tree = self._build_tree(z, r, z_grads, log_slice,
                                           direction, tree_depth-1, energy_current)

        tree_size = half_tree.size + other_half_tree.size
        sum_accept_probs = half_tree.sum_accept_probs + other_half_tree.sum_accept_probs
        num_proposals = half_tree.num_proposals + other_half_tree.num_proposals

        # Under the slice sampling process, a proposal for z is uniformly picked.
        # The probability of that proposal belongs to which half of tree
        #     is computed based on the sizes of each half.
        # For the special case that the sizes of each half are both 0,
        #     we choose the proposal from the first half
        #     (any is fine, because the probability of picking it at the end is 0!).
        if tree_size != 0:
            other_half_tree_prob = other_half_tree.size / tree_size
            is_other_half_tree = pyro.sample("is_other_halftree",
                                             dist.Bernoulli(probs=torch.ones(1) * other_half_tree_prob))
            if int(is_other_half_tree.item()) == 1:
                z_proposal = other_half_tree.z_proposal

        # leaves of the full tree are determined by the direction
        if direction == 1:
            z_left = half_tree.z_left
            r_left = half_tree.r_left
            z_left_grads = half_tree.z_left_grads
            z_right = other_half_tree.z_right
            r_right = other_half_tree.r_right
            z_right_grads = other_half_tree.z_right_grads
        else:
            z_left = other_half_tree.z_left
            r_left = other_half_tree.r_left
            z_left_grads = other_half_tree.z_left_grads
            z_right = half_tree.z_right
            r_right = half_tree.r_right
            z_right_grads = half_tree.z_right_grads

        # We already check if first half tree is turning. Now, we check
        #     if the other half tree or full tree are turning.
        turning = other_half_tree.turning or self._is_turning(z_left, r_left, z_right, r_right)

        # The divergence is checked by the second half tree (the first half is already checked).
        diverging = other_half_tree.diverging

        return _TreeInfo(z_left, r_left, z_left_grads, z_right, r_right, z_right_grads, z_proposal,
                         tree_size, turning, diverging, sum_accept_probs, num_proposals)

    def sample(self, trace):
        z = {name: node["value"].detach() for name, node in trace.iter_stochastic_nodes()}
        # automatically transform `z` to unconstrained space, if needed.
        for name, transform in self.transforms.items():
            z[name] = transform(z[name])
        r = {name: pyro.sample("r_{}_t={}".format(name, self._t), self._r_dist[name])
             for name in self._r_dist}
        energy_current = self._energy(z, r)

        # Ideally, following a symplectic integrator trajectory, the energy is constant.
        # In that case, we can sample the proposal uniformly, and there is no need to use "slice".
        # However, it is not the case for real situation: there are errors during the computation.
        # To deal with that problem, as in [1], we introduce an auxiliary "slice" variable (denoted by u).
        # The sampling process goes as follows:
        #     first sampling u from initial state (z_0, r_0) according to u ~ Uniform(0, p(z_0, r_0)),
        #     then sampling state (z, r) from the integrator trajectory according to
        #         (z, r) ~ Uniform({(z', r') in trajectory | p(z', r') >= u}).
        # For more information about slice sampling method, see
        #     `Slice sampling` by Radford M. Neal.
        # For another version of NUTS which uses multinomial sampling instead of slice sampling, see
        #     `A Conceptual Introduction to Hamiltonian Monte Carlo` by Michael Betancourt.
        joint_prob = torch.exp(-energy_current)
        if joint_prob == 0:
            slice_var = energy_current.new_tensor(0.0)
        else:
            slice_var = pyro.sample("slicevar_t={}".format(self._t),
                                    dist.Uniform(torch.zeros(1), joint_prob))
        log_slice = slice_var.log()

        z_left = z_right = z
        r_left = r_right = r
        z_left_grads = z_right_grads = None
        tree_size = 1
        accepted = False

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation.
        dist_arg_check = False if self._adapt_phase else pyro.distributions.is_validation_enabled()
        with dist.validation_enabled(dist_arg_check):
            # doubling process, stop when turning or diverging
            for tree_depth in range(self._max_tree_depth + 1):
                direction = pyro.sample("direction_t={}_treedepth={}".format(self._t, tree_depth),
                                        dist.Bernoulli(probs=torch.ones(1) * 0.5))
                direction = int(direction.item())
                if direction == 1:  # go to the right, start from the right leaf of current tree
                    new_tree = self._build_tree(z_right, r_right, z_right_grads, log_slice,
                                                direction, tree_depth, energy_current)
                    # update leaf for the next doubling process
                    z_right = new_tree.z_right
                    r_right = new_tree.r_right
                    z_right_grads = new_tree.z_right_grads
                else:  # go the the left, start from the left leaf of current tree
                    new_tree = self._build_tree(z_left, r_left, z_left_grads, log_slice,
                                                direction, tree_depth, energy_current)
                    z_left = new_tree.z_left
                    r_left = new_tree.r_left
                    z_left_grads = new_tree.z_left_grads

                if new_tree.turning or new_tree.diverging:  # stop doubling
                    break

                rand = pyro.sample("rand_t={}_treedepth={}".format(self._t, tree_depth),
                                   dist.Uniform(torch.zeros(1), torch.ones(1)))
                if rand < new_tree.size / tree_size:
                    accepted = True
                    z = new_tree.z_proposal

                if self._is_turning(z_left, r_left, z_right, r_right):  # stop doubling
                    break
                else:  # update tree_size
                    tree_size += new_tree.size

        if self._adapt_phase:
            accept_prob = new_tree.sum_accept_probs / new_tree.num_proposals
            self._adapt_step_size(accept_prob)

        if accepted:
            self._accept_cnt += 1
        self._t += 1
        # get trace with the constrained values for `z`.
        for name, transform in self.transforms.items():
            z[name] = transform.inv(z[name])
        return self._get_trace(z)
