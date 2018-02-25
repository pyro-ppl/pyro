from __future__ import absolute_import, division, print_function

from collections import defaultdict
import logging
import os

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc.mcmc import MCMC
from tests.common import assert_equal

from .test_hmc import rmse, T, TEST_CASES, TEST_IDS

logging.basicConfig(format='%(levelname)s %(message)s')
logger = logging.getLogger('pyro')
logger.setLevel(logging.INFO)

TEST_CASES[0] = TEST_CASES[0]._replace(mean_tol=0.04, std_tol=0.04)
TEST_CASES[1] = TEST_CASES[1]._replace(mean_tol=0.04, std_tol=0.04)
T2 = T(*TEST_CASES[2].values)._replace(num_samples=600, warmup_steps=100)
TEST_CASES[2] = pytest.param(*T2, marks=pytest.mark.skipif(
    'CI' in os.environ and os.environ['CI'] == 'true', reason='Slow test - skip on CI'))
T3 = T(*TEST_CASES[3].values)._replace(num_samples=700, warmup_steps=100)
TEST_CASES[3] = T3


@pytest.mark.parametrize(
    'fixture, num_samples, warmup_steps, hmc_params, expected_means, expected_precs, mean_tol, std_tol',
    TEST_CASES,
    ids=TEST_IDS)
@pytest.mark.init(rng_seed=34)
def test_nuts_conjugate_gaussian(fixture,
                                 num_samples,
                                 warmup_steps,
                                 hmc_params,
                                 expected_means,
                                 expected_precs,
                                 mean_tol,
                                 std_tol):
    nuts_kernel = NUTS(fixture.model, hmc_params['step_size'])
    mcmc_run = MCMC(nuts_kernel, num_samples, warmup_steps)
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
        expected_mean = Variable(torch.ones_like(torch.Tensor(fixture.dim)) * expected_means[i - 1])
        expected_std = 1 / torch.sqrt(Variable(torch.ones_like(torch.Tensor(fixture.dim)) * expected_precs[i - 1]))

        # Actual vs expected posterior means for the latents
        logger.info('Posterior mean (actual) - {}'.format(param_name))
        logger.info(latent_mu)
        logger.info('Posterior mean (expected) - {}'.format(param_name))
        logger.info(expected_mean)
        assert_equal(rmse(latent_mu, expected_mean).data[0], 0.0, prec=mean_tol)

        # Actual vs expected posterior precisions for the latents
        logger.info('Posterior std (actual) - {}'.format(param_name))
        logger.info(latent_std)
        logger.info('Posterior std (expected) - {}'.format(param_name))
        logger.info(expected_std)
        assert_equal(rmse(latent_std, expected_std).data[0], 0.0, prec=std_tol)


def test_logistic_regression():
    dim = 3
    true_coefs = Variable(torch.arange(1, dim+1))
    data = Variable(torch.randn(2000, dim))
    labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

    def model(data):
        coefs_mean = Variable(torch.zeros(dim), requires_grad=True)
        coefs = pyro.sample('beta', dist.Normal(coefs_mean, Variable(torch.ones(dim))))
        y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        return y

    nuts_kernel = NUTS(model, step_size=0.0855)
    mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
    posterior = []
    for trace, _ in mcmc_run._traces(data):
        posterior.append(trace.nodes['beta']['value'])
    posterior_mean = torch.mean(torch.stack(posterior), 0)
    assert_equal(rmse(true_coefs, posterior_mean).data[0], 0.0, prec=0.05)
