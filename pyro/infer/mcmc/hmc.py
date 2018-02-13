from __future__ import absolute_import, division, print_function

import numpy as np
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.mcmc.trace_kernel import TraceKernel
from pyro.ops.integrator import velocity_verlet
from pyro.util import ng_ones, ng_zeros


class HMC(TraceKernel):
    """
    Simple Hamiltonian Monte Carlo kernel, where `step_size` and `num_steps`
    need to be explicitly specified by the user.

    **Reference**
    "MCMC Using Hamiltonian Dynamics", R Neal. `pdf <https://arxiv.org/pdf/1206.1901.pdf>`_


    :param model: python callable containing pyro primitives.
    :param float step_size: determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics.
    :param int num_steps: The number of discrete steps over which to simulate
        Hamiltonian dynamics. The state at the end of the trajectory is
        returned as the proposal.
    """

    def __init__(self, model, step_size=0.5, num_steps=3):
        self.model = model
        self.step_size = step_size
        self.num_steps = num_steps
        self._reset()
        super(HMC, self).__init__()

    def _get_trace(self, z):
        z_trace = self._prototype_trace
        for name, value in z.items():
            z_trace.nodes[name]['value'] = value
        trace_poutine = poutine.trace(poutine.replay(self.model, trace=z_trace))
        trace_poutine(*self._args, **self._kwargs)
        return trace_poutine.trace

    def _potential_energy(self, z):
        return -self._get_trace(z).log_pdf()

    def _energy(self, z, r):
        kinetic_energy = 0.5 * torch.sum(torch.stack([r[name]**2 for name in r]))
        return kinetic_energy + self._potential_energy(z)

    def initial_trace(self):
        return poutine.trace(self.model).get_trace(*self._args, **self._kwargs)

    def _reset(self):
        self._t = 0
        self._r_dist = {}
        self._args = None
        self._kwargs = None
        self._accept_cnt = None
        self._prototype_trace = None

    def setup(self, *args, **kwargs):
        self._accept_cnt = 0
        self._args = args
        self._kwargs = kwargs
        # set the trace prototype to inter-convert between trace object
        # and dict object used by the integrator
        self._prototype_trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        # momenta distribution - currently standard normal
        for name, node in self._prototype_trace.iter_stochastic_nodes():
            r_mu = torch.zeros_like(node['value'])
            r_sigma = torch.ones_like(node['value'])
            self._r_dist[name] = dist.Normal(mu=r_mu, sigma=r_sigma)
        for name, node in self._prototype_trace.iter_stochastic_nodes():
            if not node['fn'].reparameterized:
                raise ValueError('Found non-reparameterized node in the model at site: {}'.format(name))
        prototype_trace_log_pdf = self._prototype_trace.log_pdf().data[0]
        if np.isnan(prototype_trace_log_pdf) or np.isinf(prototype_trace_log_pdf):
            raise ValueError('Model specification incorrect - trace log pdf is NaN, Inf or 0.')

    def cleanup(self):
        self._reset()

    def sample(self, trace):
        z = {name: node['value'] for name, node in trace.iter_stochastic_nodes()}
        r = {name: pyro.sample('r_{}_t={}'.format(name, self._t), self._r_dist[name]) for name in z}
        z_new, r_new = velocity_verlet(z, r, self._potential_energy, self.step_size, self.num_steps)
        # apply Metropolis correction
        energy_proposal = self._energy(z_new, r_new)
        energy_current = self._energy(z, r)
        delta_energy = energy_proposal - energy_current
        rand = pyro.sample('rand_t='.format(self._t), dist.Uniform(ng_zeros(1), ng_ones(1)))
        if rand.log().data[0] < -delta_energy.data[0]:
            self._accept_cnt += 1
            z = z_new
        self._t += 1
        return self._get_trace(z)

    def diagnostics(self, time_step):
        return 'Acceptance rate: {}'.format(self._accept_cnt / time_step)
