from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import torch_ones_like, torch_zeros_like
from pyro.infer import TracePosterior
from pyro.util import ng_ones, ng_zeros
from tests.common import assert_equal


def verlet_integrator(z, r, grad_potential, step_size, num_steps):
    """
    Velocity Verlet integrator.

    :param z: dictionary of sample site names and their current values
    :param r: dictionary of sample site names and corresponding momenta
    :param grad_potential: function that returns gradient of the potential given z
        for each sample site
    :return: (z_next, r_next) having same types as (z, r)
    """
    # deep copy the current state - (z, r)
    z_next = {key: val.clone().detach() for key, val in z.items()}
    r_next = {key: val.clone().detach() for key, val in r.items()}
    retain_grads(z_next)
    grads = grad_potential(z_next)

    for _ in range(num_steps):
        # detach graph nodes for next iteration
        detach_nodes(z_next)
        detach_nodes(r_next)
        for site_name in z_next:
            # r(n+1/2)
            r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
            # z(n+1)
            z_next[site_name] = z_next[site_name] + step_size * r_next[site_name]
        # retain gradients for intermediate nodes in backward step
        retain_grads(z_next)
        grads = grad_potential(z_next)
        for site_name in r_next:
            # r(n+1)
            r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
    return z_next, r_next


def retain_grads(z):
    for value in z.values():
        # XXX: can be removed with PyTorch 0.3
        if value.is_leaf and not value.requires_grad:
            value.requires_grad = True
        value.retain_grad()


def detach_nodes(z):
    for key, value in z.items():
        z[key] = Variable(value.data, requires_grad=True)


class HMC(TracePosterior):
    def __init__(self,
                 model,
                 step_size=0.5,
                 num_steps=3,
                 warmup_steps=20,
                 num_samples=1000):
        self.model = model
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        # simulation run attributes - will be set in self._setup
        # at start of run
        self._cached_param_grads = {}
        self._args = None
        self._kwargs = None
        self._accept_cnt = None
        self._prototype_trace = None
        super(HMC, self).__init__()

    def _log_prob(self, z):
        """
        Return log pdf of the model with sample sites replayed from z_trace
        """
        z_trace = self._prototype_trace.copy()
        for name, value in z.items():
            z_trace.nodes[name]['value'] = value
        model_trace = poutine.trace(poutine.replay(self.model, trace=z_trace)) \
            .get_trace(*self._args, **self._kwargs)
        return model_trace.log_pdf()

    def _grad_potential(self, z):
        log_joint_prob = self._log_prob(z)
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

    def _setup(self, *args, **kwargs):
        self._accept_cnt = 0
        self._args = args
        self._kwargs = kwargs
        # set the trace prototype to inter-convert between trace object
        # and dict object used by the integrator
        self._prototype_trace = poutine.trace(self.model).get_trace(*args, **kwargs)
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

    def _cleanup(self):
        # reset the param values to those stored before the hmc run
        for name, grad in self._cached_param_grads.items():
            param = pyro.get_param_store().get_param(name)
            param.grad = grad

    def _traces(self, *args, **kwargs):
        self._setup(*args, **kwargs)
        # sample p's from the distribution given by p_dist
        r_dist = {}
        z = {name: node['value'] for name, node in self._prototype_trace.iter_stochastic_nodes()}
        for name, value in z.items():
            r_mu = torch_zeros_like(value)
            r_sigma = torch_ones_like(value)
            r_dist[name] = dist.Normal(mu=r_mu, sigma=r_sigma)

        # Run HMC iterations
        t = 0
        print('Simulating using HMC...')
        while t < self.warmup_steps + self.num_samples:
            if t % 100 == 0:
                print('Iteration: {}'.format(t))
            # sample momentum
            r = {name: pyro.sample('r_{}_t={}'.format(name, t), r_dist[name]) for name in z}
            z_new, r_new = verlet_integrator(z,
                                             r,
                                             self._grad_potential,
                                             self.step_size,
                                             self.num_steps)
            energy_proposal = self._energy(z_new, r_new)
            energy_current = self._energy(z, r)
            # print("Energy - current: {}".format(energy_current))
            # print("Energy - proposal: {}".format(energy_proposal))
            delta_energy = energy_proposal - energy_current
            t += 1
            rand = pyro.sample('rand_t='.format(t), dist.uniform, a=ng_zeros(1), b=ng_ones(1))
            if rand.log().data[0] < -delta_energy.data[0]:
                self._accept_cnt += 1
                z = z_new
            if t < self.warmup_steps:
                continue
            yield (z, self._log_prob(z))
        self._cleanup()

    @property
    def acceptance_ratio(self):
        if self._accept_cnt is None:
            raise AttributeError("The attribute _accept_cnt is not updated until the first simulation run.")
        return self._accept_cnt / (self.warmup_steps + self.num_samples)


def test_normal_normal():
    def model(data):
        mu = pyro.param('mu', Variable(torch.zeros(10), requires_grad=True))
        x = pyro.sample('x', dist.normal, mu=mu, sigma=Variable(torch.ones(10)))
        pyro.sample('data', dist.normal, obs=data, mu=x, sigma=Variable(torch.ones(10)))

    data = Variable(torch.ones(1, 10))
    hmc = HMC(model, step_size=0.6, num_steps=3, num_samples=400, warmup_steps=50)
    traces = []
    for t, _ in hmc._traces(data):
        traces.append(t['x'])
    print('Acceptance ratio: {}'.format(hmc.acceptance_ratio))
    print('Posterior mean:')
    print(torch.mean(torch.stack(traces), 0).data)
    # gradients should not have been back-propagated.
    assert pyro.get_param_store().get_param('mu').grad is None


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
    assert_equal(q_new['x'].data[0], np.sin(1.0), prec=1.0e-4)
    assert_equal(p_new['x'].data[0], np.cos(1.0), prec=1.0e-4)
    print("q_old: {}, p_old: {}".format(q['x'].data[0], p['x'].data[0]))
    print("q_new: {}, p_new: {}".format(q_new['x'].data[0], p_new['x'].data[0]))
    print("Energy - new: {}".format(energy_new.data[0]))
    print("-------------------------------------")


def test_circular_planetary_motion():
    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.5 * p['y'] ** 2 - \
            1.0 / torch.pow(q['x'] ** 2 + q['y'] ** 2, 0.5)

    def grad(q):
        return {'x': q['x'] / torch.pow(q['x'] ** 2 + q['y'] ** 2, 1.5),
                'y': q['y'] / torch.pow(q['x'] ** 2 + q['y'] ** 2, 1.5)}

    q = {'x': Variable(torch.Tensor([1.0]), requires_grad=True),
         'y': Variable(torch.Tensor([0.0]), requires_grad=True)}
    p = {'x': Variable(torch.Tensor([0.0]), requires_grad=True),
         'y': Variable(torch.Tensor([1.0]), requires_grad=True)}
    energy_initial = energy(q, p)
    print("*** circular planetary motion ***")
    print("initial energy: {}".format(energy_initial.data[0]))
    q_new, p_new = verlet_integrator(q, p, grad, 0.01, 628)
    energy_final = energy(q_new, p_new)
    assert_equal(energy_final, energy_initial)
    assert_equal(q_new['x'].data[0], 1.0, prec=5.0e-3)
    assert_equal(q_new['y'].data[0], 0.0, prec=5.0e-3)
    print("final energy: {}".format(energy_final.data[0]))
    print("-------------------------------------")


def test_quartic_oscillator():
    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.25 * torch.pow(q['x'], 4.0)

    def grad(q):
        return {'x': torch.pow(q['x'], 3.0)}

    q = {'x': Variable(torch.Tensor([0.02]), requires_grad=True)}
    p = {'x': Variable(torch.Tensor([0.0]), requires_grad=True)}
    energy_initial = energy(q, p)
    print("*** quartic oscillator ***")
    print("initial energy: {}".format(energy_initial.data[0]))
    q_new, p_new = verlet_integrator(q, p, grad, 0.1, 1810)
    energy_final = energy(q_new, p_new)
    assert_equal(energy_final, energy_initial)
    assert_equal(q_new['x'].data[0], -0.02, prec=1.0e-4)
    assert_equal(p_new['x'].data[0], 0.0, prec=1.0e-4)
    print("final energy: {}".format(energy_final.data[0]))
    print("-------------------------------------")


def main():
    #test_verlet_integrator()
    #test_circular_planetary_motion()
    test_quartic_oscillator()
    #test_normal_normal()


if __name__ == '__main__':
    main()
