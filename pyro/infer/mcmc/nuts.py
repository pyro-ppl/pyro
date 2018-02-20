from __future__ import absolute_import, division, print_function

from collections import namedtuple

import torch

import pyro
import pyro.distributions as dist
from pyro.infer.util import torch_data_sum
from pyro.ops.integrator import velocity_verlet
from pyro.util import ng_ones, ng_zeros

from .hmc import HMC


_TreeInfo = namedtuple("TreeInfo", ["z_left", "r_left", "z_right", "r_right",
                                    "z_proposal", "size", "turning", "diverging"])


class NUTS(HMC):
    """
    No-U-Turn Sampler kernel, where ``step_size`` need to be explicitly specified by the user.

    References

    [1] `The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo`,
    Matthew D. Hoffman, and Andrew Gelman

    :param model: python callable containing pyro primitives.
    :param float step_size: determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics.
    :param int max_tree_depth: maximum depth of binary tree during No-U-turn sampling.

    Example::

        true_coefs = Variable(torch.arange(1, 4))
        data = Variable(torch.randn(2000, 3))
        labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

        def model(data):
            coefs_mean = Variable(torch.zeros(dim), requires_grad=True)
            coefs = pyro.sample('beta', dist.Normal(coefs_mean, Variable(torch.ones(3))))
            y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
            return y

        nuts_kernel = NUTS(model, step_size=0.0855)
        mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
        posterior = []
        for trace, _ in mcmc_run._traces(data):
            posterior.append(trace.nodes['beta']['value'])
    """

    def __init__(self, model, step_size=1, max_tree_depth=10):
        # The default values for step_size and max_tree_depth are selected as in Stan
        #     https://github.com/stan-dev/pystan/blob/develop/pystan/misc.py.
        super(NUTS, self).__init__(model, step_size, num_steps=None)
        self.max_tree_depth = max_tree_depth
        self._reset()

        # There are two conditions to stop doubling process:
        #     + The trajectory is making a U-turn.
        #     + The probability of the states becoming negligible: p(z, r) << u,
        # here u is the "slice" variable introduced at the `self.sample(...)` method.
        # Denote E_p = -log p(z, r), E_u = -log u, the second condition is equivalent to
        #     dE := E_p - E_u >= some constant =: dE_max.
        # This also suggests the notion "diverging" in the implemenation:
        #     when the energy E_p diverges from E_u too much, we stop doubling.
        # Here, as suggested in [1], we set dE_max = 1000.
        self._dE_max = 1000

    def _is_turning(self, z_left, r_left, z_right, r_right):
        nodes = sorted(z_left)
        z_left = torch.stack([z_left[name] for name in nodes])
        r_left = torch.stack([r_left[name] for name in nodes])
        z_right = torch.stack([z_right[name] for name in nodes])
        r_right = torch.stack([r_right[name] for name in nodes])
        dz = z_right - z_left
        return (torch_data_sum(dz * r_left) < 0) or (torch_data_sum(dz * r_right) < 0)

    def _build_basetree(self, z, r, log_slice, direction):
        step_size = self.step_size if direction == 1 else -self.step_size
        z_new, r_new = velocity_verlet(z, r, self._potential_energy, step_size)
        dE = (log_slice + self._energy(z_new, r_new)).data[0]

        # As a part of the slice sampling process (see below), along the trajectory
        #     we eliminate states which p(z, r) < u, or dE < 0.
        # Due to this elimination (and stop doubling conditions),
        #     the size of binary tree might not equal to 2^tree_depth.
        tree_size = 1 if dE <= 0 else 0
        diverging = dE >= self._dE_max
        return _TreeInfo(z_new, r_new, z_new, r_new, z_new, tree_size, False, diverging)

    def _build_tree(self, z, r, log_slice, direction, tree_depth):
        if tree_depth == 0:
            return self._build_basetree(z, r, log_slice, direction)

        # build the first half of tree
        half_tree = self._build_tree(z, r, log_slice, direction, tree_depth-1)
        z_proposal = half_tree.z_proposal

        # Check conditions to stop doubling. If we meet that condition,
        #     there is no need to build the other tree
        if half_tree.turning or half_tree.diverging:
            return half_tree

        # Else, build remaining half of tree.
        # If we are going to the right, start from the right leaf of the first half.
        if direction == 1:
            z = half_tree.z_right
            r = half_tree.r_right
        else:  # otherwise, start from the left leaf of the first half
            z = half_tree.z_left
            r = half_tree.r_left
        other_half_tree = self._build_tree(z, r, log_slice, direction, tree_depth-1)

        tree_size = half_tree.size + other_half_tree.size

        # Under the slice sampling process, a proposal for z is uniformly picked.
        # The probability of that proposal belongs to which half of tree
        #     is computed based on the sizes of each half.
        # For the special case that the sizes of each half are both 0,
        #     we choose the proposal from the first half
        #     (any is fine, because the probability of picking it at the end is 0!).
        if tree_size != 0:
            other_half_tree_prob = other_half_tree.size / tree_size
            is_other_half_tree = pyro.sample("is_other_halftree",
                                             dist.Bernoulli(ps=ng_ones(1) * other_half_tree_prob))
            if int(is_other_half_tree.data[0]) == 1:
                z_proposal = other_half_tree.z_proposal

        # leaves of the full tree is determined by the direction
        if direction == 1:
            z_left = half_tree.z_left
            r_left = half_tree.r_left
            z_right = other_half_tree.z_right
            r_right = other_half_tree.r_right
        else:
            z_left = other_half_tree.z_left
            r_left = other_half_tree.r_left
            z_right = half_tree.z_right
            r_right = half_tree.r_right

        # We already check if first half tree is turning. Now, we check
        #     if the other half tree or full tree are turning.
        turning = other_half_tree.turning or self._is_turning(z_left, r_left, z_right, r_right)

        # The divergence is checked by the second half tree (the first half is already checked).
        diverging = other_half_tree.diverging

        return _TreeInfo(z_left, r_left, z_right, r_right, z_proposal, tree_size, turning, diverging)

    def sample(self, trace):
        z = {name: node["value"] for name, node in trace.iter_stochastic_nodes()}
        r = {name: pyro.sample("r_{}_t={}".format(name, self._t), self._r_dist[name]) for name in sorted(z)}

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
        slice_var = pyro.sample("slicevar_t={}".format(self._t),
                                dist.Uniform(ng_zeros(1), torch.exp(-self._energy(z, r))))
        log_slice = slice_var.log()

        z_left = z_right = z
        r_left = r_right = r
        tree_depth = 0
        tree_size = 1
        is_accepted = False

        # doubling process, stop when turning or diverging
        for tree_depth in range(self.max_tree_depth + 1):
            direction = pyro.sample("direction_t={}_treedepth={}".format(self._t, tree_depth),
                                    dist.Bernoulli(ps=ng_ones(1) * 0.5))
            direction = int(direction.data[0])
            if direction == 1:  # go to the right, start from the right leaf of current tree
                new_tree = self._build_tree(z_right, r_right, log_slice, direction, tree_depth)
                # update leaf for the next doubling process
                z_right = new_tree.z_right
                r_right = new_tree.r_right
            else:  # go the the left, start from the left leaf of current tree
                new_tree = self._build_tree(z_left, r_left, log_slice, direction, tree_depth)
                z_left = new_tree.z_left
                r_left = new_tree.r_left

            if new_tree.turning or new_tree.diverging:  # stop doubling
                break

            accepted_prob = pyro.sample("acceptedprob_t={}_treedepth={}".format(self._t, tree_depth),
                                        dist.Uniform(ng_zeros(1), ng_ones(1)))
            if accepted_prob.data[0] < new_tree.size / tree_size:
                is_accepted = True
                z = new_tree.z_proposal

            if self._is_turning(z_left, r_left, z_right, r_right):  # stop doubling
                break
            else:  # update tree_size
                tree_size += new_tree.size

        if is_accepted:
            self._accept_cnt += 1
        self._t += 1
        return self._get_trace(z)
