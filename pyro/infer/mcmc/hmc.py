from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import eye_like
from pyro.infer import config_enumerate
from pyro.infer.mcmc.adaptation import WarmupAdapter
from pyro.infer.mcmc.trace_kernel import TraceKernel
from pyro.infer.mcmc.util import TraceEinsumEvaluator
from pyro.ops.integrator import velocity_verlet
from pyro.poutine.subsample_messenger import _Subsample
from pyro.util import optional, torch_isinf, torch_isnan, ignore_jit_warnings


class HMC(TraceKernel):
    r"""
    Simple Hamiltonian Monte Carlo kernel, where ``step_size`` and ``num_steps``
    need to be explicitly specified by the user.

    **References**

    [1] `MCMC Using Hamiltonian Dynamics`,
    Radford M. Neal

    :param model: Python callable containing Pyro primitives.
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
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer when ``jit_compile=True``. Default is False.

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
        >>> mcmc_run = MCMC(hmc_kernel, num_samples=500, warmup_steps=100).run(data)
        >>> posterior = mcmc_run.marginal('beta').empirical['beta']
        >>> posterior.mean  # doctest: +SKIP
        tensor([ 0.9819,  1.9258,  2.9737])
    """

    def __init__(self,
                 model,
                 step_size=1,
                 trajectory_length=None,
                 num_steps=None,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 full_mass=False,
                 transforms=None,
                 max_plate_nesting=None,
                 jit_compile=False,
                 ignore_jit_warnings=False):
        self.model = model
        self.max_plate_nesting = max_plate_nesting
        if trajectory_length is not None:
            self.trajectory_length = trajectory_length
        elif num_steps is not None:
            self.trajectory_length = step_size * num_steps
        else:
            self.trajectory_length = 2 * math.pi  # from Stan
        self.adapt_step_size = adapt_step_size
        self._jit_compile = jit_compile
        self._ignore_jit_warnings = ignore_jit_warnings
        self.full_mass = full_mass
        self._target_accept_prob = 0.8  # from Stan
        # The following parameter is used in find_reasonable_step_size method.
        # In NUTS paper, this threshold is set to a fixed log(0.5).
        # After https://github.com/stan-dev/stan/pull/356, it is set to a fixed log(0.8).
        self._direction_threshold = math.log(0.8)  # from Stan
        # number of tries to get a valid initial trace
        self._max_tries_initial_trace = 100
        self.transforms = {} if transforms is None else transforms
        self._automatic_transform_enabled = True if transforms is None else False
        self._reset()
        self._adapter = WarmupAdapter(step_size,
                                      adapt_step_size=adapt_step_size,
                                      adapt_mass_matrix=adapt_mass_matrix,
                                      is_diag_mass=not full_mass)
        super(HMC, self).__init__()

    def _get_trace(self, z):
        z_trace = self.initial_trace
        for name, value in z.items():
            z_trace.nodes[name]["value"] = value
        trace_poutine = poutine.trace(poutine.replay(self.model, trace=z_trace))
        trace_poutine(*self._args, **self._kwargs)
        return trace_poutine.trace

    @staticmethod
    def _iter_latent_nodes(trace):
        for name, node in sorted(trace.iter_stochastic_nodes(), key=lambda x: x[0]):
            if not (node["fn"].has_enumerate_support or isinstance(node["fn"], _Subsample)):
                yield (name, node)

    def _compute_trace_log_prob(self, model_trace):
        return self._trace_prob_evaluator.log_prob(model_trace)

    def _kinetic_energy(self, r):
        # TODO: revert to `torch.dot` in pytorch==1.0
        # See: https://github.com/uber/pyro/issues/1458
        r_flat = torch.cat([r[site_name].reshape(-1) for site_name in sorted(r)])
        if self.full_mass:
            return 0.5 * (r_flat * (self.inverse_mass_matrix.matmul(r_flat))).sum()
        else:
            return 0.5 * (self.inverse_mass_matrix * (r_flat ** 2)).sum()

    def _potential_energy(self, z):
        if self._jit_compile:
            return self._potential_energy_jit(z)
        # Since the model is specified in the constrained space, transform the
        # unconstrained R.V.s `z` to the constrained space.
        z_constrained = z.copy()
        for name, transform in self.transforms.items():
            z_constrained[name] = transform.inv(z_constrained[name])
        trace = self._get_trace(z_constrained)
        potential_energy = -self._compute_trace_log_prob(trace)
        # adjust by the jacobian for this transformation.
        for name, transform in self.transforms.items():
            potential_energy += transform.log_abs_det_jacobian(z_constrained[name], z[name]).sum()
        return potential_energy

    def _potential_energy_jit(self, z):
        names, vals = zip(*sorted(z.items()))
        if self._compiled_potential_fn:
            return self._compiled_potential_fn(*vals)

        def compiled(*zi):
            z_constrained = list(zi)
            # transform to constrained space.
            for i, name in enumerate(names):
                if name in self.transforms:
                    transform = self.transforms[name]
                    z_constrained[i] = transform.inv(z_constrained[i])
            z_constrained = dict(zip(names, z_constrained))
            trace = self._get_trace(z_constrained)
            potential_energy = -self._compute_trace_log_prob(trace)
            # adjust by the jacobian for this transformation.
            for i, name in enumerate(names):
                if name in self.transforms:
                    transform = self.transforms[name]
                    potential_energy += transform.log_abs_det_jacobian(z_constrained[name], zi[i]).sum()
            return potential_energy

        with pyro.validation_enabled(False), optional(ignore_jit_warnings(), self._ignore_jit_warnings):
            self._compiled_potential_fn = torch.jit.trace(compiled, vals, check_trace=False)
        return self._compiled_potential_fn(*vals)

    def _energy(self, z, r):
        return self._kinetic_energy(r) + self._potential_energy(z)

    def _reset(self):
        self._t = 0
        self._accept_cnt = 0
        self._r_shapes = {}
        self._r_numels = {}
        self._args = None
        self._compiled_potential_fn = None
        self._kwargs = None
        self._initial_trace = None
        self._has_enumerable_sites = False
        self._trace_prob_evaluator = None
        self._potential_energy_last = None
        self._z_grads_last = None
        self._warmup_steps = None

    def _find_reasonable_step_size(self, z):
        step_size = self.step_size

        # We are going to find a step_size which make accept_prob (Metropolis correction)
        # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
        # then we have to decrease step_size; otherwise, increase step_size.
        r, _ = self._sample_r(name="r_presample")
        energy_current = self._energy(z, r)
        z_new, r_new, z_grads, potential_energy = velocity_verlet(
            z, r, self._potential_energy, self.inverse_mass_matrix, step_size)
        energy_new = potential_energy + self._kinetic_energy(r_new)
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
        while direction_new == direction:
            step_size = step_size_scale * step_size
            z_new, r_new, z_grads, potential_energy = velocity_verlet(
                z, r, self._potential_energy, self.inverse_mass_matrix, step_size)
            energy_new = potential_energy + self._kinetic_energy(r_new)
            delta_energy = energy_new - energy_current
            direction_new = 1 if self._direction_threshold < -delta_energy else -1
        return step_size

    def _guess_max_plate_nesting(self):
        """
        Guesses max_plate_nesting by running the model once
        without enumeration. This optimistically assumes static model
        structure.
        """
        with poutine.block():
            model_trace = poutine.trace(self.model).get_trace(*self._args, **self._kwargs)
        sites = [site
                 for site in model_trace.nodes.values()
                 if site["type"] == "sample"]

        dims = [frame.dim
                for site in sites
                for frame in site["cond_indep_stack"]
                if frame.vectorized]
        self.max_plate_nesting = -min(dims) if dims else 0

    def _configure_adaptation(self):
        initial_step_size = None
        if self.adapt_step_size:
            z = {name: node["value"].detach() for name, node in self._iter_latent_nodes(self.initial_trace)}
            for name, transform in self.transforms.items():
                z[name] = transform(z[name])
            with pyro.validation_enabled(False):
                initial_step_size = self._find_reasonable_step_size(z)

        self._adapter.configure(self._warmup_steps,
                                initial_step_size)

    def _sample_r(self, name):
        r_dist = self._adapter.r_dist
        r_flat = pyro.sample(name, r_dist)
        r = {}
        pos = 0
        for name in sorted(self._r_shapes):
            next_pos = pos + self._r_numels[name]
            r[name] = r_flat[pos:next_pos].reshape(self._r_shapes[name])
            pos = next_pos
        assert pos == r_flat.size(0)
        return r, r_flat

    @property
    def inverse_mass_matrix(self):
        return self._adapter.inverse_mass_matrix

    @property
    def step_size(self):
        return self._adapter.step_size

    @property
    def num_steps(self):
        return max(1, int(self.trajectory_length / self.step_size))

    @property
    def initial_trace(self):
        """
        Find a valid trace to initiate the MCMC sampler. This is also used as a
        prototype trace to inter-convert between Pyro's trace object and dict
        object used by the integrator.
        """
        if self._initial_trace:
            return self._initial_trace
        trace = poutine.trace(self.model).get_trace(*self._args, **self._kwargs)
        for i in range(self._max_tries_initial_trace):
            trace_log_prob_sum = self._compute_trace_log_prob(trace)
            if not torch_isnan(trace_log_prob_sum) and not torch_isinf(trace_log_prob_sum):
                self._initial_trace = trace
                return trace
            trace = poutine.trace(self.model).get_trace(self._args, self._kwargs)
        raise ValueError("Model specification seems incorrect - cannot find a valid trace.")

    @initial_trace.setter
    def initial_trace(self, trace):
        self._initial_trace = trace

    def _initialize_model_properties(self):
        if self.max_plate_nesting is None:
            self._guess_max_plate_nesting()
        # Wrap model in `poutine.enum` to enumerate over discrete latent sites.
        # No-op if model does not have any discrete latents.
        self.model = poutine.enum(config_enumerate(self.model),
                                  first_available_dim=-1 - self.max_plate_nesting)
        if self._automatic_transform_enabled:
            self.transforms = {}
        trace = poutine.trace(self.model).get_trace(*self._args, **self._kwargs)
        for name, node in trace.iter_stochastic_nodes():
            if isinstance(node["fn"], _Subsample):
                continue
            if node["fn"].has_enumerate_support:
                self._has_enumerable_sites = True
                continue
            site_value = node["value"]
            if node["fn"].support is not constraints.real and self._automatic_transform_enabled:
                self.transforms[name] = biject_to(node["fn"].support).inv
                site_value = self.transforms[name](node["value"])
            self._r_shapes[name] = site_value.shape
            self._r_numels[name] = site_value.numel()
        self._trace_prob_evaluator = TraceEinsumEvaluator(trace,
                                                          self._has_enumerable_sites,
                                                          self.max_plate_nesting)
        mass_matrix_size = sum(self._r_numels.values())
        if self.full_mass:
            initial_mass_matrix = eye_like(site_value, mass_matrix_size)
        else:
            initial_mass_matrix = site_value.new_ones(mass_matrix_size)
        self._adapter.inverse_mass_matrix = initial_mass_matrix

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        self._args = args
        self._kwargs = kwargs
        self._initialize_model_properties()
        self._configure_adaptation()

    def cleanup(self):
        self._reset()

    def _cache(self, potential_energy, z_grads):
        self._potential_energy_last = potential_energy
        self._z_grads_last = z_grads

    def _fetch_from_cache(self):
        return self._potential_energy_last, self._z_grads_last

    def sample(self, trace):
        z = {name: node["value"].detach() for name, node in self._iter_latent_nodes(trace)}
        # automatically transform `z` to unconstrained space, if needed.
        for name, transform in self.transforms.items():
            z[name] = transform(z[name])

        r, _ = self._sample_r(name="r_t={}".format(self._t))

        potential_energy, z_grads = self._fetch_from_cache()
        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(z, r, self._potential_energy,
                                                                              self.inverse_mass_matrix,
                                                                              self.step_size,
                                                                              self.num_steps,
                                                                              z_grads=z_grads)
            # apply Metropolis correction.
            energy_proposal = self._kinetic_energy(r_new) + potential_energy_new
            energy_current = self._kinetic_energy(r) + potential_energy if potential_energy is not None \
                else self._energy(z, r)
        delta_energy = energy_proposal - energy_current
        # Set accept prob to 0.0 if delta_energy is `NaN` which may be
        # the case for a diverging trajectory when using a large step size.
        if torch_isnan(delta_energy):
            accept_prob = delta_energy.new_tensor(0.0)
        else:
            accept_prob = (-delta_energy).exp().clamp(max=1.)
        rand = pyro.sample("rand_t={}".format(self._t), dist.Uniform(torch.zeros(1), torch.ones(1)))
        if rand < accept_prob:
            self._accept_cnt += 1
            z = z_new

        if self._t < self._warmup_steps:
            self._adapter.step(self._t, z, accept_prob)

        self._t += 1

        # get trace with the constrained values for `z`.
        for name, transform in self.transforms.items():
            z[name] = transform.inv(z[name])
        return self._get_trace(z)

    def diagnostics(self):
        return OrderedDict([
            ("step size", "{:.2e}".format(self.step_size)),
            ("acc. rate", "{:.3f}".format(self._accept_cnt / self._t))
        ])
