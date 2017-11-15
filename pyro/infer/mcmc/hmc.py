import random

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import torch_zeros_like, torch_ones_like
from pyro.infer import TracePosterior
from pyro.poutine import Trace
from tests.common import assert_equal


def verlet_integrator(z, r, grad_potential, step_size, num_steps):
    """
    Velocity Verlet integrator.

    :param z: trace object containing current values for the sample sites
    :param r: dictionary of sample site names and corresponding momenta
    :param grad_potential: function that returns gradient of the potential given z
        for each sample site
    :return: (z_next, r_next) having same types as (z, r)
    """
    # deep copy the current state - (z, r)
    z_next = clone_trace(z)
    r_next = {k: v.clone() for k, v in r.items()}
    grads = grad_potential(z_next)
    for _ in range(num_steps):
        for site_name, node in z_next.iter_stochastic_nodes():
            # r(n+1/2)
            r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
            # z(n+1)
            node['value'] = node['value'] + step_size * r_next[site_name]
            node['value'].retain_grad()
        grads = grad_potential(z_next)
        for site_name in r_next:
            # r(n+1)
            r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
    return z_next, r_next


def clone_trace(trace):
    # clone the tensors so as not to mutate the old state (q, p)
    # if it is not accepted
    new_trace = trace.copy()
    for name, node in trace.iter_stochastic_nodes():
        node['value'].retain_grad()
        node['value'] = node['value'].clone()
    return new_trace


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
        self.cached_param_grads = {}
        self.args = None
        self.kwargs = None
        super(HMC, self).__init__()

    def log_prob(self, z_trace):
        """
        Return log pdf of the model with sample sites replayed from z_trace
        """
        model_trace = poutine.trace(poutine.replay(self.model, trace=z_trace)).get_trace(*self.args, **self.kwargs)
        return model_trace.log_pdf()

    def grad_potential(self, z):
        log_joint_prob = self.log_prob(z)
        log_joint_prob.backward()
        grad_potential = {}
        for name, node in z.iter_stochastic_nodes():
            grad_potential[name] = -node['value'].grad.clone()
            grad_potential[name].volatile = False
        return grad_potential

    def _setup(self, z):
        # store the current value of param gradients so that they
        # can be reset at the end
        for name, node in z.iter_param_nodes():
            self.cached_param_grads[name] = node['value'].grad

    def _cleanup(self):
        # reset the param values to those stored before the hmc run
        for name, grad in self.cached_param_grads.items():
            param = pyro.get_param_store().get_param(name)
            param.grad = grad

    def energy(self, z, r):
        kinetic_energy = 0.5 * torch.sum(torch.stack([r[name] ** 2 for name in r]))
        potential_energy = - self.log_prob(z)
        return kinetic_energy + potential_energy

    def _traces(self, *args, **kwargs):
        t = 0
        self.args = args
        self.kwargs = kwargs
        # initialize z with model trace
        z = poutine.trace(self.model).get_trace(*args, **kwargs)
        self._setup(z)
        # sample p's from the distribution given by p_dist
        r_dist = {}
        for name, node in z.iter_stochastic_nodes():
            r_mu = torch_zeros_like(node['value'])
            r_sigma = torch_ones_like(node['value'])
            r_dist[name] = dist.Normal(mu=r_mu, sigma=r_sigma)
        self.accept_cnt = 0

        # Run HMC iterations
        print('Simulating using HMC...')
        while t < self.warmup_steps + self.num_samples:
            if t % 100 == 0:
                print('Iteration: {}'.format(t))
            # sample momentum
            r = {name: r_dist[name].sample() for name in z.stochastic_nodes}
            z_new, r_new = verlet_integrator(z,
                                             r,
                                             self.grad_potential,
                                             self.step_size,
                                             self.num_steps)
            energy_proposal = self.energy(z_new, r_new)
            energy_current = self.energy(z, r)
            # print("Energy - current: {}".format(energy_current))
            # print("Energy - proposal: {}".format(energy_proposal))
            delta_energy = energy_proposal - energy_current
            t += 1
            if random.random() < (-delta_energy).exp().data[0]:
                self.accept_cnt += 1
                z = z_new
            if t < self.warmup_steps:
                continue
            yield (z, z.log_pdf())
        self._cleanup()

    def get_acceptance_ratio(self):
        return self.accept_cnt / float(self.warmup_steps + self.num_samples)


def test_normal_normal():
    def model(data):
        mu = pyro.param('mu', Variable(torch.zeros(10), requires_grad=True))
        x = pyro.sample('x', dist.normal, mu=mu, sigma=Variable(torch.ones(10)))
        pyro.sample('data', dist.normal, obs=data, mu=x, sigma=Variable(torch.ones(10)))

    data = Variable(torch.ones(1, 10))
    hmc = HMC(model, step_size=0.4, num_steps=3, num_samples=400, warmup_steps=50)
    traces = []
    for t, _ in hmc._traces(data):
        traces.append(t.nodes['x']['value'])
    print('Acceptance ratio: {}'.format(hmc.get_acceptance_ratio()))
    print('Posterior mean:')
    print(torch.mean(torch.stack(traces), 0).data)
    # gradients should not have been back-propagated.
    assert pyro.get_param_store().get_param('mu').grad is None


def test_verlet_integrator():
    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.5 * q.nodes['x']['value'] ** 2

    def grad(q):
        return {'x': q.nodes['x']['value']}

    q = Trace()
    q.add_node('x',
               type='sample',
               is_observed=False,
               value=Variable(torch.Tensor([0.0])))
    p = {'x': Variable(torch.Tensor([1.0]))}
    energy_cur = energy(q, p)
    print("Energy - current: {}".format(energy_cur.data[0]))
    q_new, p_new = verlet_integrator(q, p, grad, 0.01, 100)
    assert q_new.nodes['x']['value'].data[0] != q.nodes['x']['value'].data[0]
    energy_new = energy(q_new, p_new)
    assert_equal(energy_new, energy_cur)
    print("q_old: {}, p_old: {}".format(q.nodes['x']['value'].data[0], p['x'].data[0]))
    print("q_new: {}, p_new: {}".format(q_new.nodes['x']['value'].data[0], p_new['x'].data[0]))
    print("Energy - new: {}".format(energy_new.data[0]))
    print("-------------------------------------")


def main():
    test_verlet_integrator()
    test_normal_normal()


if __name__ == '__main__':
    main()
