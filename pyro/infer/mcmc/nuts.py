# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import pyro
import pyro.distributions as dist
from pyro.distributions.util import scalar_like
from pyro.infer.autoguide import init_to_uniform
from pyro.infer.mcmc.hmc import HMC
from pyro.ops.integrator import potential_grad, velocity_verlet
from pyro.util import optional, torch_isnan


def _logaddexp(x, y):
    minval, maxval = (x, y) if x < y else (y, x)
    return (minval - maxval).exp().log1p() + maxval


# sum_accept_probs and num_proposals are used to calculate
# the statistic accept_prob for Dual Averaging scheme;
# z_left_grads and z_right_grads are kept to avoid recalculating
# grads at left and right leaves;
# r_sum is used to check turning condition;
# z_proposal_pe and z_proposal_grads are used to cache the
#   potential energy and potential energy gradient values for
#   the proposal trace.
# weight is the number of valid points in case we use slice sampling
#   and is the log sum of (unnormalized) probabilites of valid points
#   when we use multinomial sampling
_TreeInfo = namedtuple("TreeInfo", ["z_left", "r_left", "r_left_unscaled", "z_left_grads",
                                    "z_right", "r_right", "r_right_unscaled", "z_right_grads",
                                    "z_proposal", "z_proposal_pe", "z_proposal_grads",
                                    "r_sum", "weight", "turning", "diverging",
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

    Refer to the `baseball example <https://github.com/pyro-ppl/pyro/blob/dev/examples/baseball.py>`_
    to see how to do Bayesian inference in Pyro using NUTS.

    **References**

    [1] `The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo`,
        Matthew D. Hoffman, and Andrew Gelman.
    [2] `A Conceptual Introduction to Hamiltonian Monte Carlo`,
        Michael Betancourt
    [3] `Slice Sampling`,
        Radford M. Neal

    :param model: Python callable containing Pyro primitives.
    :param potential_fn: Python callable calculating potential energy with input
        is a dict of real support parameters.
    :param float step_size: Determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics. If not specified, it will be set to 1.
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme.
    :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
        matrix during warm-up phase using Welford scheme.
    :param bool full_mass: A flag to decide if mass matrix is dense or diagonal.
    :param bool use_multinomial_sampling: A flag to decide if we want to sample
        candidates along its trajectory using "multinomial sampling" or using
        "slice sampling". Slice sampling is used in the original NUTS paper [1],
        while multinomial sampling is suggested in [2]. By default, this flag is
        set to True. If it is set to `False`, NUTS uses slice sampling.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is required if model contains
        discrete sample sites that can be enumerated over in parallel.
    :param bool jit_compile: Optional parameter denoting whether to use
        the PyTorch JIT to trace the log density computation, and use this
        optimized executable trace in the integrator.
    :param dict jit_options: A dictionary contains optional arguments for
        :func:`torch.jit.trace` function.
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer when ``jit_compile=True``. Default is False.
    :param float target_accept_prob: Target acceptance probability of step size
        adaptation scheme. Increasing this value will lead to a smaller step size,
        so the sampling will be slower but more robust. Default to 0.8.
    :param int max_tree_depth: Max depth of the binary tree created during the doubling
        scheme of NUTS sampler. Default to 10.
    :param callable init_strategy: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.

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
        >>> mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=300)
        >>> mcmc.run(data)
        >>> mcmc.get_samples()['beta'].mean(0)  # doctest: +SKIP
        tensor([ 0.9221,  1.9464,  2.9228])
    """

    def __init__(self,
                 model=None,
                 potential_fn=None,
                 step_size=1,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 full_mass=False,
                 use_multinomial_sampling=True,
                 transforms=None,
                 max_plate_nesting=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False,
                 target_accept_prob=0.8,
                 max_tree_depth=10,
                 init_strategy=init_to_uniform):
        super().__init__(model,
                         potential_fn,
                         step_size,
                         adapt_step_size=adapt_step_size,
                         adapt_mass_matrix=adapt_mass_matrix,
                         full_mass=full_mass,
                         transforms=transforms,
                         max_plate_nesting=max_plate_nesting,
                         jit_compile=jit_compile,
                         jit_options=jit_options,
                         ignore_jit_warnings=ignore_jit_warnings,
                         target_accept_prob=target_accept_prob,
                         init_strategy=init_strategy)
        self.use_multinomial_sampling = use_multinomial_sampling
        self._max_tree_depth = max_tree_depth
        # There are three conditions to stop doubling process:
        #     + Tree is becoming too big.
        #     + The trajectory is making a U-turn.
        #     + The probability of the states becoming negligible: p(z, r) << u,
        # here u is the "slice" variable introduced at the `self.sample(...)` method.
        # Denote E_p = -log p(z, r), E_u = -log u, the third condition is equivalent to
        #     sliced_energy := E_p - E_u > some constant =: max_sliced_energy.
        # This also suggests the notion "diverging" in the implemenation:
        #     when the energy E_p diverges from E_u too much, we stop doubling.
        # Here, as suggested in [1], we set dE_max = 1000.
        self._max_sliced_energy = 1000

    def _is_turning(self, r_left_unscaled, r_right_unscaled, r_sum):
        # We follow the strategy in Section A.4.2 of [2] for this implementation.
        left_angle = 0.
        right_angle = 0.
        for site_names, value in r_sum.items():
            rho = value - (r_left_unscaled[site_names] + r_right_unscaled[site_names]) / 2
            left_angle += r_left_unscaled[site_names].dot(rho)
            right_angle += r_right_unscaled[site_names].dot(rho)

        return (left_angle <= 0) or (right_angle <= 0)

    def _build_basetree(self, z, r, z_grads, log_slice, direction, energy_current):
        step_size = self.step_size if direction == 1 else -self.step_size
        z_new, r_new, z_grads, potential_energy = velocity_verlet(
            z, r, self.potential_fn, self.mass_matrix_adapter.kinetic_grad, step_size, z_grads=z_grads)
        r_new_unscaled = self.mass_matrix_adapter.unscale(r_new)
        energy_new = potential_energy + self._kinetic_energy(r_new_unscaled)
        # handle the NaN case
        energy_new = scalar_like(energy_new, float("inf")) if torch_isnan(energy_new) else energy_new
        sliced_energy = energy_new + log_slice
        diverging = (sliced_energy > self._max_sliced_energy)
        delta_energy = energy_new - energy_current
        accept_prob = (-delta_energy).exp().clamp(max=1.0)

        if self.use_multinomial_sampling:
            tree_weight = -sliced_energy
        else:
            # As a part of the slice sampling process (see below), along the trajectory
            #   we eliminate states which p(z, r) < u, or dE > 0.
            # Due to this elimination (and stop doubling conditions),
            #   the weight of binary tree might not equal to 2^tree_depth.
            tree_weight = scalar_like(sliced_energy, 1. if sliced_energy <= 0 else 0.)

        r_sum = r_new_unscaled
        return _TreeInfo(z_new, r_new, r_new_unscaled, z_grads, z_new, r_new, r_new_unscaled, z_grads,
                         z_new, potential_energy, z_grads, r_sum, tree_weight, False, diverging, accept_prob, 1)

    def _build_tree(self, z, r, z_grads, log_slice, direction, tree_depth, energy_current):
        if tree_depth == 0:
            return self._build_basetree(z, r, z_grads, log_slice, direction, energy_current)

        # build the first half of tree
        half_tree = self._build_tree(z, r, z_grads, log_slice,
                                     direction, tree_depth-1, energy_current)
        z_proposal = half_tree.z_proposal
        z_proposal_pe = half_tree.z_proposal_pe
        z_proposal_grads = half_tree.z_proposal_grads

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

        if self.use_multinomial_sampling:
            tree_weight = _logaddexp(half_tree.weight, other_half_tree.weight)
        else:
            tree_weight = half_tree.weight + other_half_tree.weight
        sum_accept_probs = half_tree.sum_accept_probs + other_half_tree.sum_accept_probs
        num_proposals = half_tree.num_proposals + other_half_tree.num_proposals
        r_sum = {site_names: half_tree.r_sum[site_names] + other_half_tree.r_sum[site_names]
                 for site_names in self.inverse_mass_matrix}

        # The probability of that proposal belongs to which half of tree
        #     is computed based on the weights of each half.
        if self.use_multinomial_sampling:
            other_half_tree_prob = (other_half_tree.weight - tree_weight).exp()
        else:
            # For the special case that the weights of each half are both 0,
            #   we choose the proposal from the first half
            #   (any is fine, because the probability of picking it at the end is 0!).
            other_half_tree_prob = (other_half_tree.weight / tree_weight if tree_weight > 0
                                    else scalar_like(tree_weight, 0.))
        is_other_half_tree = pyro.sample("is_other_half_tree",
                                         dist.Bernoulli(probs=other_half_tree_prob))

        if is_other_half_tree == 1:
            z_proposal = other_half_tree.z_proposal
            z_proposal_pe = other_half_tree.z_proposal_pe
            z_proposal_grads = other_half_tree.z_proposal_grads

        # leaves of the full tree are determined by the direction
        if direction == 1:
            z_left = half_tree.z_left
            r_left = half_tree.r_left
            r_left_unscaled = half_tree.r_left_unscaled
            z_left_grads = half_tree.z_left_grads
            z_right = other_half_tree.z_right
            r_right = other_half_tree.r_right
            r_right_unscaled = other_half_tree.r_right_unscaled
            z_right_grads = other_half_tree.z_right_grads
        else:
            z_left = other_half_tree.z_left
            r_left = other_half_tree.r_left
            r_left_unscaled = other_half_tree.r_left_unscaled
            z_left_grads = other_half_tree.z_left_grads
            z_right = half_tree.z_right
            r_right = half_tree.r_right
            r_right_unscaled = half_tree.r_right_unscaled
            z_right_grads = half_tree.z_right_grads

        # We already check if first half tree is turning. Now, we check
        #     if the other half tree or full tree are turning.
        turning = other_half_tree.turning or self._is_turning(r_left_unscaled, r_right_unscaled, r_sum)

        # The divergence is checked by the second half tree (the first half is already checked).
        diverging = other_half_tree.diverging

        return _TreeInfo(z_left, r_left, r_left_unscaled, z_left_grads, z_right, r_right, r_right_unscaled,
                         z_right_grads, z_proposal, z_proposal_pe, z_proposal_grads, r_sum, tree_weight,
                         turning, diverging, sum_accept_probs, num_proposals)

    def sample(self, params):
        z, potential_energy, z_grads = self._fetch_from_cache()
        # recompute PE when cache is cleared
        if z is None:
            z = params
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
            self._cache(z, potential_energy, z_grads)
        # return early if no sample sites
        elif len(z) == 0:
            self._t += 1
            self._mean_accept_prob = 1.
            if self._t > self._warmup_steps:
                self._accept_cnt += 1
            return z
        r, r_unscaled = self._sample_r(name="r_t={}".format(self._t))
        energy_current = self._kinetic_energy(r_unscaled) + potential_energy

        # Ideally, following a symplectic integrator trajectory, the energy is constant.
        # In that case, we can sample the proposal uniformly, and there is no need to use "slice".
        # However, it is not the case for real situation: there are errors during the computation.
        # To deal with that problem, as in [1], we introduce an auxiliary "slice" variable (denoted
        # by u).
        # The sampling process goes as follows:
        #   first sampling u from initial state (z_0, r_0) according to
        #     u ~ Uniform(0, p(z_0, r_0)),
        #   then sampling state (z, r) from the integrator trajectory according to
        #     (z, r) ~ Uniform({(z', r') in trajectory | p(z', r') >= u}).
        #
        # For more information about slice sampling method, see [3].
        # For another version of NUTS which uses multinomial sampling instead of slice sampling,
        # see [2].

        if self.use_multinomial_sampling:
            log_slice = -energy_current
        else:
            # Rather than sampling the slice variable from `Uniform(0, exp(-energy))`, we can
            # sample log_slice directly using `energy`, so as to avoid potential underflow or
            # overflow issues ([2]).
            slice_exp_term = pyro.sample("slicevar_exp_t={}".format(self._t),
                                         dist.Exponential(scalar_like(energy_current, 1.)))
            log_slice = -energy_current - slice_exp_term

        z_left = z_right = z
        r_left = r_right = r
        r_left_unscaled = r_right_unscaled = r_unscaled
        z_left_grads = z_right_grads = z_grads
        accepted = False
        r_sum = r_unscaled
        sum_accept_probs = 0.
        num_proposals = 0
        tree_weight = scalar_like(energy_current, 0. if self.use_multinomial_sampling else 1.)

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation.
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            # doubling process, stop when turning or diverging
            tree_depth = 0
            while tree_depth < self._max_tree_depth:
                direction = pyro.sample("direction_t={}_treedepth={}".format(self._t, tree_depth),
                                        dist.Bernoulli(probs=scalar_like(tree_weight, 0.5)))
                direction = int(direction.item())
                if direction == 1:  # go to the right, start from the right leaf of current tree
                    new_tree = self._build_tree(z_right, r_right, z_right_grads, log_slice,
                                                direction, tree_depth, energy_current)
                    # update leaf for the next doubling process
                    z_right = new_tree.z_right
                    r_right = new_tree.r_right
                    r_right_unscaled = new_tree.r_right_unscaled
                    z_right_grads = new_tree.z_right_grads
                else:  # go the the left, start from the left leaf of current tree
                    new_tree = self._build_tree(z_left, r_left, z_left_grads, log_slice,
                                                direction, tree_depth, energy_current)
                    z_left = new_tree.z_left
                    r_left = new_tree.r_left
                    r_left_unscaled = new_tree.r_left_unscaled
                    z_left_grads = new_tree.z_left_grads

                sum_accept_probs = sum_accept_probs + new_tree.sum_accept_probs
                num_proposals = num_proposals + new_tree.num_proposals

                # stop doubling
                if new_tree.diverging:
                    if self._t >= self._warmup_steps:
                        self._divergences.append(self._t - self._warmup_steps)
                    break

                if new_tree.turning:
                    break

                tree_depth += 1

                if self.use_multinomial_sampling:
                    new_tree_prob = (new_tree.weight - tree_weight).exp()
                else:
                    new_tree_prob = new_tree.weight / tree_weight
                rand = pyro.sample("rand_t={}_treedepth={}".format(self._t, tree_depth),
                                   dist.Uniform(scalar_like(new_tree_prob, 0.),
                                                scalar_like(new_tree_prob, 1.)))
                if rand < new_tree_prob:
                    accepted = True
                    z = new_tree.z_proposal
                    z_grads = new_tree.z_proposal_grads
                    self._cache(z, new_tree.z_proposal_pe, z_grads)

                r_sum = {site_names: r_sum[site_names] + new_tree.r_sum[site_names]
                         for site_names in r_unscaled}
                if self._is_turning(r_left_unscaled, r_right_unscaled, r_sum):  # stop doubling
                    break
                else:  # update tree_weight
                    if self.use_multinomial_sampling:
                        tree_weight = _logaddexp(tree_weight, new_tree.weight)
                    else:
                        tree_weight = tree_weight + new_tree.weight

        accept_prob = sum_accept_probs / num_proposals

        self._t += 1
        if self._t > self._warmup_steps:
            n = self._t - self._warmup_steps
            if accepted:
                self._accept_cnt += 1
        else:
            n = self._t
            self._adapter.step(self._t, z, accept_prob, z_grads)
        self._mean_accept_prob += (accept_prob.item() - self._mean_accept_prob) / n

        return z.copy()
