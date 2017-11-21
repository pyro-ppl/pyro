import logging
from collections import defaultdict

import pytest
import torch
import pyro.distributions as dist
from torch.autograd import Variable

import pyro
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.mcmc.verlet_integrator import verlet_integrator
from tests.common import assert_equal


logging.basicConfig(format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GaussianChain(object):
    def __init__(self, dim, n, mu_0, lambda_prec):
        self.dim = dim
        self.n = n
        self.mu_0 = Variable(torch.Tensor(torch.ones(self.dim) * mu_0), requires_grad=True)
        self.lambda_prec = Variable(torch.Tensor(torch.ones(self.dim) * lambda_prec))

    def model(self, data):
        mu = pyro.param('mu_0', self.mu_0)
        lambda_prec = self.lambda_prec
        for i in range(1, self.n + 1):
            mu = pyro.sample('mu_{}'.format(i), dist.normal, mu=mu, sigma=Variable(lambda_prec.data))
        pyro.sample('obs', dist.normal, mu=mu, sigma=Variable(lambda_prec.data), obs=data)

    def analytic_means(self, data):
        lambda_tilde_posts = [self.lambda_prec]
        for k in range(1, self.n):
            lambda_tilde_k = (self.lambda_prec * lambda_tilde_posts[k - 1]) /\
                (self.lambda_prec + lambda_tilde_posts[k - 1])
            lambda_tilde_posts.append(lambda_tilde_k)
        lambda_n_post = data.size()[0] * self.lambda_prec + lambda_tilde_posts[self.n - 1]

        target_mus = [None] * self.n
        target_mu_n = data.sum(dim=0) * self.lambda_prec / lambda_n_post +\
            self.mu_0 * lambda_tilde_posts[self.n - 1] / lambda_n_post
        target_mus[-1] = target_mu_n
        for k in range(self.n-2, -1, -1):
            target_mus[k] = (self.mu_0 * lambda_tilde_posts[k - 1] + target_mus[k+1] * self.lambda_prec) / \
                            (self.lambda_prec + lambda_tilde_posts[k])
        return target_mus


class TestFixture(object):
    def __init__(self, dim, chain_len, num_obs, step_size, num_steps, num_samples=600):
        self.dim = dim
        self.chain_len = chain_len
        self.num_obs = num_obs
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.fixture = GaussianChain(dim, chain_len, 0, 1)

    @property
    def model(self):
        return self.fixture.model

    @property
    def data(self):
        return Variable(torch.ones(self.num_obs, self.dim))

    def analytic_means(self, data):
        return self.fixture.analytic_means(data)

    def id_fn(self):
        return 'dim={}_chain-len={}_num_obs={}'.format(self.dim, self.chain_len, self.num_obs)


def mse(t1, t2):
    return (t1 - t2).pow(2).mean()


@pytest.mark.parametrize('fixture', [
    TestFixture(dim=10, chain_len=3, num_obs=1, step_size=0.5, num_steps=4, num_samples=600),
    TestFixture(dim=10, chain_len=3, num_obs=5, step_size=0.4, num_steps=3, num_samples=700),
    TestFixture(dim=10, chain_len=7, num_obs=1, step_size=0.4, num_steps=4, num_samples=1300),
    ], ids=lambda x: x.id_fn())
def test_hmc_conjugate_gaussian(fixture):
    mcmc_run = MCMC(fixture.model,
                    kernel=HMC,
                    num_samples=fixture.num_samples,
                    warmup_steps=50,
                    step_size=fixture.step_size,
                    num_steps=fixture.num_steps)
    post_trace = defaultdict(list)
    for t, _ in mcmc_run._traces(fixture.data):
        for i in range(1, fixture.chain_len+1):
            param_name = 'mu_' + str(i)
            post_trace[param_name].append(t.nodes[param_name]['value'])
    analytic_means = fixture.analytic_means(fixture.data)
    logger.info('Acceptance ratio: {}'.format(mcmc_run.acceptance_ratio))
    for i in range(1, fixture.chain_len+1):
        param_name = 'mu_' + str(i)
        latent_mu = torch.mean(torch.stack(post_trace[param_name]), 0)
        analytic_mean = analytic_means[i-1]
        # Actual vs expected posterior means for the latents
        logger.info('Posterior - {}'.format(param_name))
        logger.info(latent_mu)
        logger.info('Posterior (analytic) - {}'.format(param_name))
        logger.info(analytic_means[i-1])
        assert_equal(mse(latent_mu, analytic_mean).data[0], 0, prec=0.01)


def test_verlet_integrator():
    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.5 * q['x'] ** 2

    def grad(q):
        return {'x': q['x']}

    q = {'x': Variable(torch.Tensor([0.0]), requires_grad=True)}
    p = {'x': Variable(torch.Tensor([1.0]), requires_grad=True)}
    energy_cur = energy(q, p)
    logger.info("Energy - current: {}".format(energy_cur.data[0]))
    q_new, p_new = verlet_integrator(q, p, grad, 0.01, 100)
    assert q_new['x'].data[0] != q['x'].data[0]
    energy_new = energy(q_new, p_new)
    assert_equal(energy_new, energy_cur)
    logger.info("q_old: {}, p_old: {}".format(q['x'].data[0], p['x'].data[0]))
    logger.info("q_new: {}, p_new: {}".format(q_new['x'].data[0], p_new['x'].data[0]))
    logger.info("Energy - new: {}".format(energy_new.data[0]))
