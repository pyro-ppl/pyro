from __future__ import absolute_import, division, print_function

import logging
from collections import defaultdict, namedtuple

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC
from tests.common import assert_equal

logging.basicConfig(format='%(levelname)s %(message)s')
logging.getLogger('pyro').setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class GaussianChain(object):

    def __init__(self, dim, chain_len, num_obs):
        self.dim = dim
        self.chain_len = chain_len
        self.num_obs = num_obs
        self.mu_0 = Variable(torch.zeros(self.dim), requires_grad=True)
        self.lambda_prec = Variable(torch.ones(self.dim))

    def model(self, data):
        mu = pyro.param('mu_0', self.mu_0)
        lambda_prec = self.lambda_prec
        for i in range(1, self.chain_len + 1):
            mu = pyro.sample('mu_{}'.format(i), dist.normal, mu=mu, sigma=Variable(lambda_prec.data))
        pyro.sample('obs', dist.normal, mu=mu, sigma=Variable(lambda_prec.data), obs=data)

    @property
    def data(self):
        return Variable(torch.ones(self.num_obs, self.dim))

    def id_fn(self):
        return 'dim={}_chain-len={}_num_obs={}'.format(self.dim, self.chain_len, self.num_obs)


def rmse(t1, t2):
    return (t1 - t2).pow(2).mean().sqrt()


TestCase = namedtuple('TestCase', [
    'fixture',
    'num_samples',
    'warmup_steps',
    'hmc_params',
    'expected_means',
    'expected_precs',
    'mean_tol',
    'std_tol'])

# TODO: delete slow test cases once code is in master
TEST_CASES = [
    TestCase(GaussianChain(dim=10, chain_len=3, num_obs=1),
             num_samples=800,
             warmup_steps=50,
             hmc_params={'step_size': 0.5,
                         'num_steps': 4},
             expected_means=[0.25, 0.50, 0.75],
             expected_precs=[1.33, 1, 1.33],
             mean_tol=0.03,
             std_tol=0.06),
    # XXX: Very sensitive to HMC parameters. Biased estimate is obtained
    # without enough samples and/or larger step size.
    TestCase(GaussianChain(dim=5, chain_len=2, num_obs=10000),
             num_samples=2000,
             warmup_steps=500,
             hmc_params={'step_size': 0.013,
                         'num_steps': 25},
             expected_means=[0.5, 1.0],
             expected_precs=[2.0, 10000],
             mean_tol=0.05,
             std_tol=0.05),
    TestCase(GaussianChain(dim=10, chain_len=4, num_obs=1),
             num_samples=1200,
             warmup_steps=300,
             hmc_params={'step_size': 0.46,
                         'num_steps': 5},
             expected_means=[0.20, 0.40, 0.60, 0.80],
             expected_precs=[1.25, 0.83, 0.83, 1.25],
             mean_tol=0.06,
             std_tol=0.06),
    TestCase(GaussianChain(dim=5, chain_len=9, num_obs=1),
             num_samples=3000,
             warmup_steps=500,
             hmc_params={'step_size': 0.3,
                         'num_steps': 8},
             expected_means=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
             expected_precs=[1.11, 0.63, 0.48, 0.42, 0.4, 0.42, 0.48, 0.63, 1.11],
             mean_tol=0.08,
             std_tol=0.08),
]

TEST_IDS = [t[0].id_fn() for t in TEST_CASES]


@pytest.mark.parametrize(
    'fixture, num_samples, warmup_steps, hmc_params, expected_means, expected_precs, mean_tol, std_tol',
    TEST_CASES,
    ids=TEST_IDS)
def test_hmc_conjugate_gaussian(fixture,
                                num_samples,
                                warmup_steps,
                                hmc_params,
                                expected_means,
                                expected_precs,
                                mean_tol,
                                std_tol):
    hmc_kernel = HMC(fixture.model, **hmc_params)
    mcmc_run = MCMC(hmc_kernel, num_samples, warmup_steps)
    pyro.get_param_store().clear()
    post_trace = defaultdict(list)
    for t, _ in mcmc_run._traces(fixture.data):
        for i in range(1, fixture.chain_len + 1):
            param_name = 'mu_' + str(i)
            post_trace[param_name].append(t.nodes[param_name]['value'])
    for i in range(1, fixture.chain_len + 1):
        param_name = 'mu_' + str(i)
        latent_mu = torch.mean(torch.stack(post_trace[param_name]), 0)
        latent_std = torch.std(torch.stack(post_trace[param_name]), 0)
        expected_mean = Variable(torch.ones(fixture.dim) * expected_means[i - 1])
        expected_std = 1 / torch.sqrt(Variable(torch.ones(fixture.dim) * expected_precs[i - 1]))

        # Actual vs expected posterior means for the latents
        logger.info('Posterior mean (actual) - {}'.format(param_name))
        logger.info(latent_mu)
        logger.info('Posterior mean (expected) - {}'.format(param_name))
        logger.info(expected_mean)
        assert_equal(rmse(latent_mu, expected_mean).data[0], 0, prec=mean_tol)

        # Actual vs expected posterior precisions for the latents
        logger.info('Posterior std (actual) - {}'.format(param_name))
        logger.info(latent_std)
        logger.info('Posterior std (expected) - {}'.format(param_name))
        logger.info(expected_std)
        assert_equal(rmse(latent_std, expected_std).data[0], 0, prec=std_tol)
