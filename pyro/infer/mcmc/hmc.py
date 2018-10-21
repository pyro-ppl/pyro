from __future__ import absolute_import, division, print_function

import math
import warnings
from collections import OrderedDict

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import eye_like
from pyro.infer import config_enumerate
from pyro.infer.mcmc.trace_kernel import TraceKernel
from pyro.infer.mcmc.util import TraceEinsumEvaluator, TraceTreeEvaluator
from pyro.ops.dual_averaging import DualAveraging
from pyro.ops.integrator import single_step_velocity_verlet, velocity_verlet
from pyro.ops.welford import WelfordCovariance
from pyro.poutine.subsample_messenger import _Subsample
from pyro.util import optional, torch_isinf, torch_isnan


class HMC(TraceKernel):
    """
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
    :param bool experimental_use_einsum: Whether to use an einsum operation
        to evaluate log pdf for the model trace. No-op unless the trace has
        discrete sample sites. This flag is experimental and will most likely
        be removed in a future release.

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
        >>> posterior = EmpiricalMarginal(mcmc_run, 'beta')
        >>> posterior.mean  # doctest: +SKIP
        tensor([ 0.9819,  1.9258,  2.9737])
    """

    def __init__(self,
                 model,
                 step_size=None,
                 trajectory_length=None,
                 num_steps=None,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 full_mass=False,
                 transforms=None,
                 max_plate_nesting=float("inf"),
                 max_iarange_nesting=None,  # DEPRECATED
                 experimental_use_einsum=False):
        self.model = model
        if max_iarange_nesting is not None:
            warnings.warn("max_iarange_nesting is deprecated; use max_plate_nesting instead",
                          DeprecationWarning)
            max_plate_nesting = max_iarange_nesting
        self.max_plate_nesting = max_plate_nesting
        self.step_size = step_size if step_size is not None else 1  # from Stan
        if trajectory_length is not None:
            self.trajectory_length = trajectory_length
        elif num_steps is not None:
            self.trajectory_length = self.step_size * num_steps
        else:
            self.trajectory_length = 2 * math.pi  # from Stan
        self.num_steps = max(1, int(self.trajectory_length / self.step_size))
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.full_mass = full_mass
        self.use_einsum = experimental_use_einsum
        self._target_accept_prob = 0.8  # from Stan
        # The following parameter is used in find_reasonable_step_size method.
        # In NUTS paper, this threshold is set to a fixed log(0.5).
        # After https://github.com/stan-dev/stan/pull/356, it is set to a fixed log(0.8).
        self._direction_threshold = math.log(0.8)  # from Stan

        # We separate warmup_steps into windows:
        #   start_buffer + window 1 + window 2 + window 3 + ... + end_buffer
        # where the length of each window will be doubled for the next window.
        # We won't adapt mass matrix during start and end buffers; and mass
        # matrix will be updated at the end of each window. This is helpful
        # for dealing with the intense computation of sampling momentum from the
        # inverse of mass matrix.
        self._adapt_start_buffer = 75  # from Stan
        self._adapt_end_buffer = 50  # from Stan
        self._adapt_initial_window = 25  # from Stan

        # number of tries to get a valid prototype trace
        self._max_tries_prototype_trace = 100

        self.transforms = {} if transforms is None else transforms
        self._automatic_transform_enabled = True if transforms is None else False
        self._reset()
        super(HMC, self).__init__()

    def _get_trace(self, z):
        z_trace = self._prototype_trace
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
            return 0.5 * (r_flat * (self._inverse_mass_matrix.matmul(r_flat))).sum()
        else:
            return 0.5 * (self._inverse_mass_matrix * (r_flat ** 2)).sum()

    def _potential_energy(self, z):
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

    def _energy(self, z, r):
        return self._kinetic_energy(r) + self._potential_energy(z)

    def _reset(self):
        self._t = 0
        self._accept_cnt = 0
        self._inverse_mass_matrix = None
        self._r_dist = None
        self._r_shapes = {}
        self._r_numels = {}
        self._args = None
        self._kwargs = None
        self._prototype_trace = None
        self._adapt_phase = False
        self._adapt_mass_matrix_phase = False
        self._step_size_adapt_scheme = None
        self._mass_matrix_adapt_scheme = None
        self._has_enumerable_sites = False
        self._trace_prob_evaluator = None

    def _find_reasonable_step_size(self, z):
        step_size = self.step_size

        # We are going to find a step_size which make accept_prob (Metropolis correction)
        # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
        # then we have to decrease step_size; otherwise, increase step_size.
        r, _ = self._sample_r(name="r_presample")
        energy_current = self._energy(z, r)
        z_new, r_new, z_grads, potential_energy = single_step_velocity_verlet(
            z, r, self._potential_energy, self._inverse_mass_matrix, step_size)
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
            z_new, r_new, z_grads, potential_energy = single_step_velocity_verlet(
                z, r, self._potential_energy, self._inverse_mass_matrix, step_size)
            energy_new = potential_energy + self._kinetic_energy(r_new)
            delta_energy = energy_new - energy_current
            direction_new = 1 if self._direction_threshold < -delta_energy else -1
        return step_size

    def _configure_adaptation(self, trace):
        self._adapt_phase = True
        self._adapt_window = self._adapt_initial_window

        if self.adapt_step_size:
            z = {name: node["value"].detach() for name, node in self._iter_latent_nodes(trace)}
            for name, transform in self.transforms.items():
                z[name] = transform(z[name])
            with pyro.validation_enabled(False):
                self.step_size = self._find_reasonable_step_size(z)
            self.num_steps = max(1, int(self.trajectory_length / self.step_size))
            # make prox-center for Dual Averaging scheme
            loc = math.log(10 * self.step_size)
            self._step_size_adapt_scheme = DualAveraging(prox_center=loc)

        # from Stan, for small warmup_steps
        if self._warmup_steps < 20:
            self._adapt_window_ending = self._warmup_steps
            return

        if (self._adapt_start_buffer + self._adapt_end_buffer
                + self._adapt_initial_window > self._warmup_steps):
            self._adapt_start_buffer = int(0.15 * self._warmup_steps)
            self._adapt_end_buffer = int(0.1 * self._warmup_steps)
            self._adapt_window = (self._warmup_steps - self._adapt_start_buffer
                                  - self._adapt_end_buffer)

        # define ending pointer of the current window
        self._adapt_window_ending = self._adapt_start_buffer
        self._adapt_mass_matrix_phase_ending = self._warmup_steps - self._adapt_end_buffer

        if self.adapt_mass_matrix:
            is_diag = not self.full_mass
            self._mass_matrix_adapt_scheme = WelfordCovariance(diagonal=is_diag)

    def _end_warmup(self):
        self._adapt_phase = False
        if self.adapt_step_size:
            _, log_step_size_avg = self._step_size_adapt_scheme.get_state()
            self.step_size = math.exp(log_step_size_avg)
            self.num_steps = max(1, int(self.trajectory_length / self.step_size))

    def _end_adapt_window(self):
        if self._adapt_window_ending == self._warmup_steps:
            self._end_warmup()
            return

        if self._adapt_window_ending == self._adapt_start_buffer:
            if self.adapt_mass_matrix:
                self._adapt_mass_matrix_phase = True
        else:
            if self.adapt_step_size:
                self._step_size_adapt_scheme.prox_center = math.log(10 * self.step_size)
                self._step_size_adapt_scheme.reset()

            if self.adapt_mass_matrix:
                self._inverse_mass_matrix = self._mass_matrix_adapt_scheme.get_covariance()
                self._update_r_dist()
                self._mass_matrix_adapt_scheme.reset()

            self._adapt_window = 2 * self._adapt_window
            if self._adapt_window_ending == self._adapt_mass_matrix_phase_ending:
                self._adapt_mass_matrix_phase = False
                self._adapt_window_ending = self._warmup_steps
                return

        self._adapt_window_ending = self._adapt_window_ending + self._adapt_window
        # expanding the current window if length of the next one is too large
        next_adapt_window_ending = self._adapt_window_ending + 2 * self._adapt_window
        if next_adapt_window_ending > self._adapt_mass_matrix_phase_ending:
            self._adapt_window_ending = self._adapt_mass_matrix_phase_ending

    def _adapt_step_size(self, accept_prob):
        # calculate a statistic for Dual Averaging scheme
        H = self._target_accept_prob - accept_prob
        self._step_size_adapt_scheme.step(H)
        log_step_size, _ = self._step_size_adapt_scheme.get_state()
        self.step_size = math.exp(log_step_size)
        self.num_steps = max(1, int(self.trajectory_length / self.step_size))

    def _update_r_dist(self):
        loc = self._inverse_mass_matrix.new_zeros(self._inverse_mass_matrix.size(0))
        if self.full_mass:
            self._r_dist = dist.MultivariateNormal(loc,
                                                   precision_matrix=self._inverse_mass_matrix)
        else:
            self._r_dist = dist.Normal(loc, self._inverse_mass_matrix.rsqrt())

    def _sample_r(self, name):
        r_flat = pyro.sample(name, self._r_dist)
        r = {}
        pos = 0
        for name in sorted(self._r_shapes):
            next_pos = pos + self._r_numels[name]
            r[name] = r_flat[pos:next_pos].reshape(self._r_shapes[name])
            pos = next_pos
        assert pos == r_flat.size(0)
        return r, r_flat

    def _set_valid_prototype_trace(self, trace):
        trace_eval = TraceEinsumEvaluator if self.use_einsum else TraceTreeEvaluator
        self._trace_prob_evaluator = trace_eval(trace,
                                                self._has_enumerable_sites,
                                                self.max_plate_nesting)
        for i in range(self._max_tries_prototype_trace):
            trace_log_prob_sum = self._compute_trace_log_prob(trace)
            if not torch_isnan(trace_log_prob_sum) and not torch_isinf(trace_log_prob_sum):
                self._prototype_trace = trace
                return
            trace = poutine.trace(self.model).get_trace(self._args, self._kwargs)
        raise ValueError("Model specification seems incorrect - can not find a valid trace.")

    def initial_trace(self):
        return self._prototype_trace

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        self._args = args
        self._kwargs = kwargs
        # Wrap model in `poutine.enum` to enumerate over discrete latent sites.
        # No-op if model does not have any discrete latents.
        self.model = poutine.enum(config_enumerate(self.model, default="parallel"),
                                  first_available_dim=self.max_plate_nesting)
        # set the trace prototype to inter-convert between trace object
        # and dict object used by the integrator
        trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        if self._automatic_transform_enabled:
            self.transforms = {}
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

        self._set_valid_prototype_trace(trace)
        mass_matrix_size = sum(self._r_numels.values())
        if self.full_mass:
            self._inverse_mass_matrix = eye_like(site_value, mass_matrix_size)
        else:
            self._inverse_mass_matrix = site_value.new_ones(mass_matrix_size)
        self._update_r_dist()
        if self.adapt_step_size or self.adapt_mass_matrix:
            self._configure_adaptation(trace)

    def cleanup(self):
        self._reset()

    def sample(self, trace):
        z = {name: node["value"].detach() for name, node in self._iter_latent_nodes(trace)}
        # automatically transform `z` to unconstrained space, if needed.
        for name, transform in self.transforms.items():
            z[name] = transform(z[name])

        r, _ = self._sample_r(name="r_t={}".format(self._t))

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation
        with optional(pyro.validation_enabled(False), self._adapt_phase):
            z_new, r_new = velocity_verlet(z, r,
                                           self._potential_energy,
                                           self._inverse_mass_matrix,
                                           self.step_size,
                                           self.num_steps)
            # apply Metropolis correction.
            energy_proposal = self._energy(z_new, r_new)
            energy_current = self._energy(z, r)
        delta_energy = energy_proposal - energy_current
        rand = pyro.sample("rand_t={}".format(self._t), dist.Uniform(torch.zeros(1), torch.ones(1)))
        if rand < (-delta_energy).exp():
            self._accept_cnt += 1
            z = z_new

        self._t += 1

        if self._adapt_phase:
            if self.adapt_step_size:
                # Set accept prob to 0.0 if delta_energy is `NaN` which may be
                # the case for a diverging trajectory when using a large step size.
                if torch_isnan(delta_energy):
                    accept_prob = delta_energy.new_tensor(0.0)
                else:
                    accept_prob = (-delta_energy).exp().clamp(max=1).item()
                self._adapt_step_size(accept_prob)
            if self._adapt_mass_matrix_phase:
                z_flat = torch.cat([z[name].reshape(-1) for name in sorted(z)])
                self._mass_matrix_adapt_scheme.update(z_flat.detach())
            if self._t == self._adapt_window_ending:
                self._end_adapt_window()

        # get trace with the constrained values for `z`.
        for name, transform in self.transforms.items():
            z[name] = transform.inv(z[name])
        return self._get_trace(z)

    def diagnostics(self):
        return OrderedDict([
            ("Step size", self.step_size),
            ("Acceptance rate", self._accept_cnt / self._t)
        ])
