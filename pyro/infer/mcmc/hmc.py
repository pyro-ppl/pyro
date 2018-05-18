from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.mcmc.trace_kernel import TraceKernel
from pyro.ops.dual_averaging import DualAveraging
from pyro.ops.integrator import single_step_velocity_verlet, velocity_verlet
from pyro.util import torch_isinf, torch_isnan


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
        >>> hmc_kernel = HMC(model, step_size=0.0855, num_steps=4)
        >>> mcmc_run = MCMC(hmc_kernel, num_samples=500, warmup_steps=100).run(data)
        >>> posterior = EmpiricalMarginal(mcmc_run, 'beta')
        >>> posterior.mean  # doctest: +SKIP
        tensor([ 0.9819,  1.9258,  2.9737])
    """

    def __init__(self, model, step_size=None, trajectory_length=None,
                 num_steps=None, adapt_step_size=False, transforms=None):
        self.model = model

        self.step_size = step_size if step_size is not None else 1  # from Stan
        if trajectory_length is not None:
            self.trajectory_length = trajectory_length
        elif num_steps is not None:
            self.trajectory_length = self.step_size * num_steps
        else:
            self.trajectory_length = 2 * math.pi  # from Stan
        self.num_steps = max(1, int(self.trajectory_length / self.step_size))
        self.adapt_step_size = adapt_step_size
        self._target_accept_prob = 0.8  # from Stan

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

    def _kinetic_energy(self, r):
        return 0.5 * sum(x.pow(2).sum() for x in r.values())

    def _potential_energy(self, z):
        # Since the model is specified in the constrained space, transform the
        # unconstrained R.V.s `z` to the constrained space.
        z_constrained = z.copy()
        for name, transform in self.transforms.items():
            z_constrained[name] = transform.inv(z_constrained[name])
        trace = self._get_trace(z_constrained)
        potential_energy = -trace.log_prob_sum()
        # adjust by the jacobian for this transformation.
        for name, transform in self.transforms.items():
            potential_energy += transform.log_abs_det_jacobian(z_constrained[name], z[name]).sum()
        return potential_energy

    def _energy(self, z, r):
        return self._kinetic_energy(r) + self._potential_energy(z)

    def _reset(self):
        self._t = 0
        self._accept_cnt = 0
        self._r_dist = OrderedDict()
        self._args = None
        self._kwargs = None
        self._prototype_trace = None
        self._adapt_phase = False
        self._adapted_scheme = None

    def _find_reasonable_step_size(self, z):
        step_size = self.step_size
        # NOTE: This target_accept_prob is 0.5 in NUTS paper, is 0.8 in Stan,
        # and is different to the target_accept_prob for Dual Averaging scheme.
        # We need to discuss which one is better.
        target_accept_logprob = math.log(self._target_accept_prob)

        # We are going to find a step_size which make accept_prob (Metropolis correction)
        # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
        # then we have to decrease step_size; otherwise, increase step_size.
        r = {name: pyro.sample("r_{}_presample".format(name), self._r_dist[name])
             for name in self._r_dist}
        energy_current = self._energy(z, r)
        z_new, r_new, z_grads, potential_energy = single_step_velocity_verlet(
            z, r, self._potential_energy, step_size)
        energy_new = potential_energy + self._kinetic_energy(r_new)
        delta_energy = energy_new - energy_current
        # direction=1 means keep increasing step_size, otherwise decreasing step_size.
        # Note that the direction is -1 if delta_energy is `NaN` which may be the
        # case for a diverging trajectory (e.g. in the case of evaluating log prob
        # of a value simulated using a large step size for a constrained sample
        # site).
        direction = 1 if target_accept_logprob < -delta_energy else -1

        # define scale for step_size: 2 for increasing, 1/2 for decreasing
        step_size_scale = 2 ** direction
        direction_new = direction
        # keep scale step_size until accept_prob crosses its target
        # TODO: make thresholds for too small step_size or too large step_size
        while direction_new == direction:
            step_size = step_size_scale * step_size
            z_new, r_new, z_grads, potential_energy = single_step_velocity_verlet(
                z, r, self._potential_energy, step_size)
            energy_new = potential_energy + self._kinetic_energy(r_new)
            delta_energy = energy_new - energy_current
            direction_new = 1 if target_accept_logprob < -delta_energy else -1
        return step_size

    def _adapt_step_size(self, accept_prob):
        # calculate a statistic for Dual Averaging scheme
        H = self._target_accept_prob - accept_prob
        self._adapted_scheme.step(H)
        log_step_size, _ = self._adapted_scheme.get_state()
        self.step_size = math.exp(log_step_size)
        self.num_steps = max(1, int(self.trajectory_length / self.step_size))

    def _validate_trace(self, trace):
        trace_log_prob_sum = trace.log_prob_sum()
        if torch_isnan(trace_log_prob_sum) or torch_isinf(trace_log_prob_sum):
            raise ValueError("Model specification incorrect - trace log pdf is NaN or Inf.")

    def initial_trace(self):
        return self._prototype_trace

    def setup(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        # set the trace prototype to inter-convert between trace object
        # and dict object used by the integrator
        trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        self._prototype_trace = trace
        if self._automatic_transform_enabled:
            self.transforms = {}
        for name, node in sorted(trace.iter_stochastic_nodes(), key=lambda x: x[0]):
            site_value = node["value"]
            if node["fn"].support is not constraints.real and self._automatic_transform_enabled:
                self.transforms[name] = biject_to(node["fn"].support).inv
                site_value = self.transforms[name](node["value"])
            r_loc = site_value.new_zeros(site_value.shape)
            r_scale = site_value.new_ones(site_value.shape)
            self._r_dist[name] = dist.Normal(loc=r_loc, scale=r_scale)
        self._validate_trace(trace)

        if self.adapt_step_size:
            self._adapt_phase = True
            z = {name: node["value"] for name, node in trace.iter_stochastic_nodes()}
            for name, transform in self.transforms.items():
                z[name] = transform(z[name])
            self.step_size = self._find_reasonable_step_size(z)
            self.num_steps = max(1, int(self.trajectory_length / self.step_size))
            # make prox-center for Dual Averaging scheme
            loc = math.log(10 * self.step_size)
            self._adapted_scheme = DualAveraging(prox_center=loc)

    def end_warmup(self):
        if self.adapt_step_size:
            self._adapt_phase = False
            _, log_step_size_avg = self._adapted_scheme.get_state()
            self.step_size = math.exp(log_step_size_avg)
            self.num_steps = max(1, int(self.trajectory_length / self.step_size))

    def cleanup(self):
        self._reset()

    def sample(self, trace):
        z = {name: node["value"].detach() for name, node in trace.iter_stochastic_nodes()}
        # automatically transform `z` to unconstrained space, if needed.
        for name, transform in self.transforms.items():
            z[name] = transform(z[name])
        r = {name: pyro.sample("r_{}_t={}".format(name, self._t), self._r_dist[name])
             for name in self._r_dist}

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation
        dist_arg_check = False if self._adapt_phase else pyro.distributions.is_validation_enabled()
        with dist.validation_enabled(dist_arg_check):
            z_new, r_new = velocity_verlet(z, r,
                                           self._potential_energy,
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

        if self._adapt_phase:
            # Set accept prob to 0.0 if delta_energy is `NaN` which may be
            # the case for a diverging trajectory when using a large step size.
            if torch_isnan(delta_energy):
                accept_prob = delta_energy.new_tensor(0.0)
            else:
                accept_prob = (-delta_energy).exp().clamp(max=1).item()
            self._adapt_step_size(accept_prob)

        self._t += 1
        # get trace with the constrained values for `z`.
        for name, transform in self.transforms.items():
            z[name] = transform.inv(z[name])
        return self._get_trace(z)

    def diagnostics(self):
        return "Step size: {:.6f} \t Acceptance rate: {:.6f}".format(
            self.step_size, self._accept_cnt / self._t)
