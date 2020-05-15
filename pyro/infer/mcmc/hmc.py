# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.util import scalar_like
from pyro.distributions.testing.fakes import NonreparameterizedNormal

from pyro.infer.autoguide import init_to_uniform
from pyro.infer.mcmc.adaptation import WarmupAdapter
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import potential_grad, velocity_verlet
from pyro.util import optional, torch_isnan


class HMC(MCMCKernel):
    r"""
    Simple Hamiltonian Monte Carlo kernel, where ``step_size`` and ``num_steps``
    need to be explicitly specified by the user.

    **References**

    [1] `MCMC Using Hamiltonian Dynamics`,
    Radford M. Neal

    :param model: Python callable containing Pyro primitives.
    :param potential_fn: Python callable calculating potential energy with input
        is a dict of real support parameters.
    :param float step_size: Determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics. If not specified, it will be set to 1.
    :param float trajectory_length: Length of a MCMC trajectory. If not
        specified, it will be set to ``step_size x num_steps``. In case
        ``num_steps`` is not specified, it will be set to :math:`2\pi`.
    :param int num_steps: The number of discrete steps over which to simulate
        Hamiltonian dynamics. The state at the end of the trajectory is
        returned as the proposal. This value is always equal to
        ``int(trajectory_length / step_size)``.
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme.
    :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
        matrix during warm-up phase using Welford scheme.
    :param bool full_mass: A flag to decide if mass matrix is dense or diagonal.
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
    :param float target_accept_prob: Increasing this value will lead to a smaller
        step size, hence the sampling will be slower and more robust. Default to 0.8.
    :param callable init_strategy: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.

    .. note:: Internally, the mass matrix will be ordered according to the order
        of the names of latent variables, not the order of their appearance in
        the model.

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
        >>> hmc_kernel = HMC(model, step_size=0.0855, num_steps=4)
        >>> mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100)
        >>> mcmc.run(data)
        >>> mcmc.get_samples()['beta'].mean(0)  # doctest: +SKIP
        tensor([ 0.9819,  1.9258,  2.9737])
    """

    def __init__(self,
                 model=None,
                 potential_fn=None,
                 step_size=1,
                 trajectory_length=None,
                 num_steps=None,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 full_mass=False,
                 transforms=None,
                 max_plate_nesting=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False,
                 target_accept_prob=0.8,
                 init_strategy=init_to_uniform):
        if not ((model is None) ^ (potential_fn is None)):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        # NB: deprecating args - model, transforms
        self.model = model
        self.transforms = transforms
        self._max_plate_nesting = max_plate_nesting
        self._jit_compile = jit_compile
        self._jit_options = jit_options
        self._ignore_jit_warnings = ignore_jit_warnings
        self._init_strategy = init_strategy

        self.potential_fn = potential_fn
        if trajectory_length is not None:
            self.trajectory_length = trajectory_length
        elif num_steps is not None:
            self.trajectory_length = step_size * num_steps
        else:
            self.trajectory_length = 2 * math.pi  # from Stan
        # The following parameter is used in find_reasonable_step_size method.
        # In NUTS paper, this threshold is set to a fixed log(0.5).
        # After https://github.com/stan-dev/stan/pull/356, it is set to a fixed log(0.8).
        self._direction_threshold = math.log(0.8)  # from Stan
        self._max_sliced_energy = 1000
        self._reset()
        self._adapter = WarmupAdapter(step_size,
                                      adapt_step_size=adapt_step_size,
                                      adapt_mass_matrix=adapt_mass_matrix,
                                      target_accept_prob=target_accept_prob,
                                      dense_mass=full_mass)
        super().__init__()

    def _kinetic_energy(self, r_unscaled):
        energy = 0.
        for site_names, value in r_unscaled.items():
            energy = energy + value.dot(value)
        return 0.5 * energy

    def _reset(self):
        self._t = 0
        self._accept_cnt = 0
        self._mean_accept_prob = 0.
        self._divergences = []
        self._prototype_trace = None
        self._initial_params = None
        self._z_last = None
        self._potential_energy_last = None
        self._z_grads_last = None
        self._warmup_steps = None

    def _find_reasonable_step_size(self, z):
        step_size = self.step_size

        # We are going to find a step_size which make accept_prob (Metropolis correction)
        # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
        # then we have to decrease step_size; otherwise, increase step_size.
        potential_energy = self.potential_fn(z)
        r, r_unscaled = self._sample_r(name="r_presample_0")
        energy_current = self._kinetic_energy(r_unscaled) + potential_energy
        # This is required so as to avoid issues with autograd when model
        # contains transforms with cache_size > 0 (https://github.com/pyro-ppl/pyro/issues/2292)
        z = {k: v.clone() for k, v in z.items()}
        z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(
            z, r, self.potential_fn, self.mass_matrix_adapter.kinetic_grad, step_size)
        r_new_unscaled = self.mass_matrix_adapter.unscale(r_new)
        energy_new = self._kinetic_energy(r_new_unscaled) + potential_energy_new
        delta_energy = energy_new - energy_current
        # direction=1 means keep increasing step_size, otherwise decreasing step_size.
        # Note that the direction is -1 if delta_energy is `NaN` which may be the
        # case for a diverging trajectory (e.g. in the case of evaluating log prob
        # of a value simulated using a large step size for a constrained sample site).
        direction = 1 if self._direction_threshold < -delta_energy else -1

        # define scale for step_size: 2 for increasing, 1/2 for decreasing
        step_size_scale = 2 ** direction
        direction_new = direction
        # keep scale step_size until accept_prob crosses its target
        # TODO: make thresholds for too small step_size or too large step_size
        t = 0
        while direction_new == direction:
            t += 1
            step_size = step_size_scale * step_size
            r, r_unscaled = self._sample_r(name="r_presample_{}".format(t))
            energy_current = self._kinetic_energy(r_unscaled) + potential_energy
            z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(
                z, r, self.potential_fn, self.mass_matrix_adapter.kinetic_grad, step_size)
            r_new_unscaled = self.mass_matrix_adapter.unscale(r_new)
            energy_new = self._kinetic_energy(r_new_unscaled) + potential_energy_new
            delta_energy = energy_new - energy_current
            direction_new = 1 if self._direction_threshold < -delta_energy else -1
        return step_size

    def _sample_r(self, name):
        r_unscaled = {}
        options = {"dtype": self._potential_energy_last.dtype,
                   "device": self._potential_energy_last.device}
        for site_names, size in self.mass_matrix_adapter.mass_matrix_size.items():
            # we want to sample from Normal distribution using `sample` method rather than
            # `rsample` method because the former is a bit faster
            r_unscaled[site_names] = pyro.sample(
                "{}_{}".format(name, site_names),
                NonreparameterizedNormal(torch.zeros(size, **options), torch.ones(size, **options)))

        r = self.mass_matrix_adapter.scale(r_unscaled, r_prototype=self.initial_params)
        return r, r_unscaled

    @property
    def mass_matrix_adapter(self):
        return self._adapter.mass_matrix_adapter

    @mass_matrix_adapter.setter
    def mass_matrix_adapter(self, value):
        self._adapter.mass_matrix_adapter = value

    @property
    def inverse_mass_matrix(self):
        return self.mass_matrix_adapter.inverse_mass_matrix

    @property
    def step_size(self):
        return self._adapter.step_size

    @property
    def num_steps(self):
        return max(1, int(self.trajectory_length / self.step_size))

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def _initialize_model_properties(self, model_args, model_kwargs):
        init_params, potential_fn, transforms, trace = initialize_model(
            self.model,
            model_args,
            model_kwargs,
            transforms=self.transforms,
            max_plate_nesting=self._max_plate_nesting,
            jit_compile=self._jit_compile,
            jit_options=self._jit_options,
            skip_jit_warnings=self._ignore_jit_warnings,
            init_strategy=self._init_strategy,
            initial_params=self._initial_params,
        )
        self.potential_fn = potential_fn
        self.transforms = transforms
        self._initial_params = init_params
        self._prototype_trace = trace

    def _initialize_adapter(self):
        if self._adapter.dense_mass is False:
            dense_sites_list = []
        elif self._adapter.dense_mass is True:
            dense_sites_list = [tuple(sorted(self.initial_params))]
        else:
            msg = "full_mass should be a list of tuples of site names."
            dense_sites_list = self._adapter.dense_mass
            assert isinstance(dense_sites_list, list), msg
            for dense_sites in dense_sites_list:
                assert dense_sites and isinstance(dense_sites, tuple), msg
                for name in dense_sites:
                    assert isinstance(name, str) and name in self.initial_params, msg
        dense_sites_set = set().union(*dense_sites_list)
        diag_sites = tuple(sorted([name for name in self.initial_params
                                   if name not in dense_sites_set]))
        assert len(diag_sites) + sum([len(sites) for sites in dense_sites_list]) == len(self.initial_params), \
            "Site names specified in full_mass are duplicated."

        mass_matrix_shape = OrderedDict()
        for dense_sites in dense_sites_list:
            size = sum([self.initial_params[site].numel() for site in dense_sites])
            mass_matrix_shape[dense_sites] = (size, size)

        if diag_sites:
            size = sum([self.initial_params[site].numel() for site in diag_sites])
            mass_matrix_shape[diag_sites] = (size,)

        options = {"dtype": self._potential_energy_last.dtype,
                   "device": self._potential_energy_last.device}
        self._adapter.configure(self._warmup_steps,
                                mass_matrix_shape=mass_matrix_shape,
                                find_reasonable_step_size_fn=self._find_reasonable_step_size,
                                options=options)

        if self._adapter.adapt_step_size:
            self._adapter.reset_step_size_adaptation(self._initial_params)

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)
        if self.initial_params:
            z = {k: v.detach() for k, v in self.initial_params.items()}
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
        else:
            z_grads, potential_energy = {}, self.potential_fn(self.initial_params)
        self._cache(self.initial_params, potential_energy, z_grads)
        if self.initial_params:
            self._initialize_adapter()

    def cleanup(self):
        self._reset()

    def _cache(self, z, potential_energy, z_grads=None):
        self._z_last = z
        self._potential_energy_last = potential_energy
        self._z_grads_last = z_grads

    def clear_cache(self):
        self._z_last = None
        self._potential_energy_last = None
        self._z_grads_last = None

    def _fetch_from_cache(self):
        return self._z_last, self._potential_energy_last, self._z_grads_last

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
            return params
        r, r_unscaled = self._sample_r(name="r_t={}".format(self._t))
        energy_current = self._kinetic_energy(r_unscaled) + potential_energy

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(
                z, r, self.potential_fn, self.mass_matrix_adapter.kinetic_grad,
                self.step_size, self.num_steps, z_grads=z_grads)
            # apply Metropolis correction.
            r_new_unscaled = self.mass_matrix_adapter.unscale(r_new)
            energy_proposal = self._kinetic_energy(r_new_unscaled) + potential_energy_new
        delta_energy = energy_proposal - energy_current
        # handle the NaN case which may be the case for a diverging trajectory
        # when using a large step size.
        delta_energy = scalar_like(delta_energy, float("inf")) if torch_isnan(delta_energy) else delta_energy
        if delta_energy > self._max_sliced_energy and self._t >= self._warmup_steps:
            self._divergences.append(self._t - self._warmup_steps)

        accept_prob = (-delta_energy).exp().clamp(max=1.)
        rand = pyro.sample("rand_t={}".format(self._t), dist.Uniform(scalar_like(accept_prob, 0.),
                                                                     scalar_like(accept_prob, 1.)))
        accepted = False
        if rand < accept_prob:
            accepted = True
            z = z_new
            z_grads = z_grads_new
            self._cache(z, potential_energy_new, z_grads)

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

    def logging(self):
        return OrderedDict([
            ("step size", "{:.2e}".format(self.step_size)),
            ("acc. prob", "{:.3f}".format(self._mean_accept_prob))
        ])

    def diagnostics(self):
        return {"divergences": self._divergences,
                "acceptance rate": self._accept_cnt / (self._t - self._warmup_steps)}
