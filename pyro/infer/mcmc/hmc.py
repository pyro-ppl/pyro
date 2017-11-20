from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import torch_ones_like, torch_zeros_like
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.mcmc.trace_kernel import TraceKernel
from pyro.infer.mcmc.verlet_integrator import verlet_integrator
from pyro.util import ng_ones, ng_zeros
from tests.common import assert_equal


class HMC(TraceKernel):
    def __init__(self,
                 model,
                 step_size=0.5,
                 num_steps=3):
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
        print(log_joint_prob)
        log_joint_prob.backward()
        grad_potential = {}
        for name, value in z.items():
            grad_potential[name] = -value.grad.clone().detach()
            grad_potential[name].volatile = False
        return grad_potential

    def _energy(self, z, r):
        kinetic_energy = 0.5 * torch.sum(torch.stack([r[name] ** 2 for name in r]))
        potential_energy = - self._log_prob(z)
        return kinetic_energy + potential_energy

    def setup(self, *args, **kwargs):
        self._accept_cnt = 0
        self._args = args
        self._kwargs = kwargs
        # set the trace prototype to inter-convert between trace object
        # and dict object used by the integrator
        self._prototype_trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        for name, node in self._prototype_trace.iter_stochastic_nodes():
            r_mu = torch_zeros_like(node['value'])
            r_sigma = torch_ones_like(node['value'])
            self._r_dist[name] = dist.Normal(mu=r_mu, sigma=r_sigma)
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
        z_new, r_new = verlet_integrator(z,
                                         r,
                                         self._grad_potential,
                                         self.step_size,
                                         self.num_steps)
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


def test_normal_normal():
    def model(data):
        mu = pyro.param('mu', Variable(torch.zeros(10), requires_grad=True))
        x = pyro.sample('x', dist.normal, mu=mu, sigma=Variable(torch.ones(10)))
        y = pyro.sample('y', dist.normal, mu=x, sigma=Variable(torch.ones(10)))
        pyro.sample('data', dist.normal, obs=data, mu=y, sigma=Variable(torch.ones(10)))

    data = Variable(torch.ones(1, 10))
    hmc = MCMC(model, kernel=HMC, num_samples=400, warmup_steps=50, step_size=0.6, num_steps=3)
    traces = []
    for t, _ in hmc._traces(data):
        traces.append(t['x'])
    print('Acceptance ratio: {}'.format(hmc.acceptance_ratio))
    print('Posterior mean:')
    print(torch.mean(torch.stack(traces), 0).data)
    # gradients should not have been back-propagated.
    print(pyro.get_param_store().get_param('mu').grad)


def test_verlet_integrator():
    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.5 * q['x'] ** 2

    def grad(q):
        return {'x': q['x']}

    q = {'x': Variable(torch.Tensor([0.0]), requires_grad=True)}
    p = {'x': Variable(torch.Tensor([1.0]), requires_grad=True)}
    energy_cur = energy(q, p)
    print("Energy - current: {}".format(energy_cur.data[0]))
    q_new, p_new = verlet_integrator(q, p, grad, 0.01, 100)
    assert q_new['x'].data[0] != q['x'].data[0]
    energy_new = energy(q_new, p_new)
    assert_equal(energy_new, energy_cur)
    print("q_old: {}, p_old: {}".format(q['x'].data[0], p['x'].data[0]))
    print("q_new: {}, p_new: {}".format(q_new['x'].data[0], p_new['x'].data[0]))
    print("Energy - new: {}".format(energy_new.data[0]))
    print("-------------------------------------")


def main():
    # test_verlet_integrator()
    test_normal_normal()


if __name__ == '__main__':
    main()
