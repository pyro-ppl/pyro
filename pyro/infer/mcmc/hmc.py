from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import math

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.mcmc.trace_kernel import TraceKernel
from pyro.ops.dual_averaging import DualAveraging
from pyro.ops.integrator import velocity_verlet, single_step_velocity_verlet
from pyro.util import ng_ones, ng_zeros, is_nan, is_inf


class HMC(TraceKernel):
    """
    Simple Hamiltonian Monte Carlo kernel, where ``step_size`` and ``num_steps``
    need to be explicitly specified by the user.

    References

    [1] `MCMC Using Hamiltonian Dynamics`,
    Radford M. Neal

    :param model: python callable containing pyro primitives.
    :param float step_size: determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics.
    :param int num_steps: The number of discrete steps over which to simulate
        Hamiltonian dynamics. The state at the end of the trajectory is
        returned as the proposal.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.
    """

    def __init__(self, model, step_size=0.5, num_steps=3, transforms=None):
        self.model = model
        self.step_size = step_size
        self.num_steps = num_steps
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
        return 0.5 * torch.sum(torch.stack([r[name]**2 for name in r]))

    def _potential_energy(self, z):
        # Since the model is specified in the constrained space, transform the
        # unconstrained R.V.s `z` to the constrained space.
        z_constrained = z.copy()
        for name, transform in self.transforms.items():
            z_constrained[name] = transform.inv(z_constrained[name])
        trace = self._get_trace(z_constrained)
        potential_energy = -trace.log_pdf()
        # adjust by the jacobian for this transformation.
        for name, transform in self.transforms.items():
            potential_energy += transform.log_abs_det_jacobian(z_constrained[name], z[name]).sum()
        return potential_energy

    def _energy(self, z, r):
        return self._kinetic_energy(r) + self._potential_energy(z)

    def _reset(self):
        self._t = 0
        self._accept_cnt = 0
        # TODO: move these parameters to self.defaults dict or self.config dict of init method
        self._target_accept_prob = 0.8  # from Stan
        self._adapted_step_size = self.step_size if self.step_size is not None else 1
        self._trajectory_length = 2 * math.pi  # from Stan
        self._t0 = 10
        self._kappa = 0.75
        self._gamma = 0.05
        self._adapted = False
        self._warmup = True

        self._r_dist = OrderedDict()
        self._args = None
        self._kwargs = None
        self._prototype_trace = None
        self._adapted_scheme = None

    def _find_reasonable_step_size(self, z):
        step_size = self._adapted_step_size
        # This target_accept_prob is 0.5 in NUTS paper, is 0.8 in Stan, and
        # is different to the target_accept_prob for Dual Averaging scheme.
        # We need to discuss which one is better.
        target_accept_logprob = math.log(self._target_accept_prob)

        r = {name: pyro.sample("r_{}_presample".format(name), self._r_dist[name])
             for name in self._r_dist}
        energy_current = self._energy(z, r)
        z_new, r_new, z_grads, potential_energy = single_step_velocity_verlet(
            z, r, self._potential_energy, step_size)
        energy_new = potential_energy + self._kinetic_energy(r_new)
        delta_energy = energy_new - energy_current
        direction = 1 if target_accept_logprob < -delta_energy else -1
        # if accept_prob:=exp(-delta_energy) is small, then we have to
        # decrease step_size; otherwise, increase step_size
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
        if self._warmup:
            # calculate the statistics for Dual Averaging scheme
            H = self._target_accept_prob - accept_prob
            self._adapted_scheme.step(H)
            log_step_size, _ = self._adapted_scheme.get_state()
            self._adapted_step_size = math.exp(log_step_size)
        else:
            _, log_step_size_avg = self._adapted_scheme.get_state()
            self._adapted_step_size = math.exp(log_step_size_avg)

    def _validate_trace(self, trace):
        trace_log_pdf = trace.log_pdf()
        if is_nan(trace_log_pdf) or is_inf(trace_log_pdf):
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
        # momenta distribution - currently standard normal
        for name, node in sorted(trace.iter_stochastic_nodes(), key=lambda x: x[0]):
            r_mu = torch.zeros_like(node["value"])
            r_sigma = torch.ones_like(node["value"])
            self._r_dist[name] = dist.Normal(mu=r_mu, sigma=r_sigma)
        if node["fn"].support is not constraints.real and self._automatic_transform_enabled:
            self.transforms[name] = biject_to(node["fn"].support).inv
        self._validate_trace(trace)

        if self._adapted:
            z = {name: node["value"] for name, node in trace.iter_stochastic_nodes()}
            for name, transform in self.transforms.items():
                z[name] = transform(z[name])
            self._adapted_step_size = self._find_reasonable_step_size(z)
            mu = math.log(10 * self._adapted_step_size)
            self._adapted_scheme = DualAveraging(mu, self._t0, self._kappa, self._gamma)

    def cleanup(self):
        self._reset()

    def sample(self, trace):
        z = {name: node["value"] for name, node in trace.iter_stochastic_nodes()}
        # automatically transform `z` to unconstrained space, if needed.
        for name, transform in self.transforms.items():
            z[name] = transform(z[name])
        r = {name: pyro.sample("r_{}_t={}".format(name, self._t), self._r_dist[name])
             for name in self._r_dist}

        if self._adapted:
            step_size = self._adapted_step_size
            num_steps = int(self._trajectory_length / self._adapted_step_size)
        else:
            step_size = self.step_size
            num_steps = self.num_steps
        z_new, r_new = velocity_verlet(z, r,
                                       self._potential_energy,
                                       step_size,
                                       num_steps)
        # apply Metropolis correction.
        energy_proposal = self._energy(z_new, r_new)
        energy_current = self._energy(z, r)
        delta_energy = energy_proposal - energy_current
        rand = pyro.sample("rand_t={}".format(self._t), dist.Uniform(ng_zeros(1), ng_ones(1)))
        if rand < (-delta_energy).exp():
            self._accept_cnt += 1
            z = z_new

        if self._adapted:
            accept_prob = (-delta_energy).exp().clamp(max=1).item()
            self._adapt_step_size(accept_prob)

        self._t += 1
        # get trace with the constrained values for `z`.
        for name, transform in self.transforms.items():
            z[name] = transform.inv(z[name])
        return self._get_trace(z)

    def diagnostics(self):
        return "Step size: {:.06f} | Acceptance rate: {:.06f}".format(
            self._adapted_step_size, self._accept_cnt / self._t)
