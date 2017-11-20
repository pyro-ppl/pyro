import logging

import pytest
import torch
import pyro.distributions as dist
from torch.autograd import Variable

import pyro
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC

logger = logging.getLogger()
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

    def analytic_values(self, data):
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


@pytest.mark.parametrize('dim, chain_len, num_samples', [(10, 2, 1)])
def test_hmc_conj_gaussian(dim, chain_len, num_samples):
    fixture = GaussianChain(dim, chain_len, 0, 1)
    data = Variable(torch.ones(num_samples, dim))
    mcmc_run = MCMC(fixture.model, kernel=HMC, num_samples=600, warmup_steps=50, step_size=0.5, num_steps=4)
    traces = []
    for t, _ in mcmc_run._traces(data):
        traces.append(t.nodes['mu_2']['value'])
    print('Acceptance ratio: {}'.format(mcmc_run.acceptance_ratio))
    print('Posterior mean:')
    print(torch.mean(torch.stack(traces), 0).data)


# g = GaussianChain(2, 2, 0, 1)
# data = Variable(torch.ones(1, 2))
# print g.analytic_values(data)