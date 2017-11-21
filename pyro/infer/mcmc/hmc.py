from __future__ import absolute_import, division, print_function

import numpy as np
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import torch_ones_like, torch_zeros_like
from pyro.infer.mcmc.trace_kernel import TraceKernel
from pyro.infer.mcmc.verlet_integrator import verlet_integrator
from pyro.util import ng_ones, ng_zeros


class HMC(TraceKernel):

    def __init__(self, model, step_size=0.5, num_steps=3):
        self.model = model
        self.step_size = step_size
        self.num_steps = num_steps
        # simulation run attributes - will be set in self._setup
        # at start of run
        self._r_dist = {}
        self._cached_param_grads = {}
        self._args = None
        self._kwargs = None
        self._accept_cnt = None
        self._prototype_trace = None
        super(HMC, self).__init__()

    def _get_trace(self, z):
        z_trace = self._prototype_trace.copy()
        for name, value in z.items():
            z_trace.nodes[name]['value'] = value
        return poutine.trace(poutine.replay(self.model, trace=z_trace)) \
            .get_trace(*self._args, **self._kwargs)

    def _log_prob(self, z):
        """
        Return log pdf of the model with sample sites replayed from z_trace
        """
        return self._get_trace(z).log_pdf()

    def _grad_potential(self, z):
        log_joint_prob = self._log_prob(z)
        log_joint_prob.backward()
        grad_potential = {}
        for name, value in z.items():
            grad_potential[name] = -value.grad.clone().detach()
            grad_potential[name].volatile = False
        return grad_potential

    def _energy(self, z, r):
        kinetic_energy = 0.5 * torch.sum(torch.stack([r[name]**2 for name in r]))
        potential_energy = -self._log_prob(z)
        return kinetic_energy + potential_energy

    def setup(self, *args, **kwargs):
        self._accept_cnt = 0
        self._args = args
        self._kwargs = kwargs
        # set the trace prototype to inter-convert between trace object
        # and dict object used by the integrator
        self._prototype_trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        # momenta distribution - currently standard normal
        for name, node in self._prototype_trace.iter_stochastic_nodes():
            r_mu = torch_zeros_like(node['value'])
            r_sigma = torch_ones_like(node['value'])
            self._r_dist[name] = dist.Normal(mu=r_mu, sigma=r_sigma)
        # validate model
        for name, node in self._prototype_trace.iter_stochastic_nodes():
            if not node['fn'].reparameterized:
                raise ValueError('Found non-reparameterized node in the model at site: {}'.format(name))
        prototype_trace_log_pdf = self._prototype_trace.log_pdf().data[0]
        if np.isnan(prototype_trace_log_pdf) or np.isinf(prototype_trace_log_pdf):
            raise ValueError('Model specification incorrect - trace log pdf is NaN, Inf or 0.')
        # store the current value of param gradients so that they
        # can be reset at the end
        for name, node in self._prototype_trace.iter_param_nodes():
            self._cached_param_grads[name] = node['value'].grad

    def cleanup(self):
        # reset the param values to those stored before the hmc run
        for name, grad in self._cached_param_grads.items():
            param = pyro.get_param_store().get_param(name)
            param.grad = grad

    def sample(self, trace, time_step):
        z = {name: node['value'] for name, node in trace.iter_stochastic_nodes()}
        # sample p's from the distribution given by p_dist
        # sample momentum
        r = {name: pyro.sample('r_{}_t={}'.format(name, time_step), self._r_dist[name]) for name in z}
        z_new, r_new = verlet_integrator(z, r, self._grad_potential, self.step_size, self.num_steps)
        energy_proposal = self._energy(z_new, r_new)
        energy_current = self._energy(z, r)
        delta_energy = energy_proposal - energy_current
        rand = pyro.sample('rand_t='.format(time_step), dist.uniform, a=ng_zeros(1), b=ng_ones(1))
        if rand.log().data[0] < -delta_energy.data[0]:
            self._accept_cnt += 1
            z = z_new
        return self._get_trace(z)

    @property
    def num_accepts(self):
        return self._accept_cnt
