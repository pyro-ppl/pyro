from __future__ import absolute_import, division, print_function

import logging
import os

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS
from tests.common import assert_equal

from .test_hmc import TEST_CASES, TEST_IDS, T, rmse

logger = logging.getLogger(__name__)

T2 = T(*TEST_CASES[2].values)._replace(num_samples=800, warmup_steps=200)
TEST_CASES[2] = pytest.param(*T2, marks=pytest.mark.skipif(
    'CI' in os.environ and os.environ['CI'] == 'true', reason='Slow test - skip on CI'))
T3 = T(*TEST_CASES[3].values)._replace(num_samples=700, warmup_steps=100)
TEST_CASES[3] = T3


@pytest.mark.parametrize(
    'fixture, num_samples, warmup_steps, hmc_params, expected_means, expected_precs, mean_tol, std_tol',
    TEST_CASES,
    ids=TEST_IDS)
@pytest.mark.init(rng_seed=34)
@pytest.mark.disable_validation()
def test_nuts_conjugate_gaussian(fixture,
                                 num_samples,
                                 warmup_steps,
                                 hmc_params,
                                 expected_means,
                                 expected_precs,
                                 mean_tol,
                                 std_tol):
    pyro.get_param_store().clear()
    nuts_kernel = NUTS(fixture.model, hmc_params['step_size'])
    mcmc_run = MCMC(nuts_kernel, num_samples, warmup_steps).run(fixture.data)
    for i in range(1, fixture.chain_len + 1):
        param_name = 'loc_' + str(i)
        marginal = EmpiricalMarginal(mcmc_run, sites=param_name)
        latent_loc = marginal.mean
        latent_std = marginal.variance.sqrt()
        expected_mean = torch.ones(fixture.dim) * expected_means[i - 1]
        expected_std = 1 / torch.sqrt(torch.ones(fixture.dim) * expected_precs[i - 1])

        # Actual vs expected posterior means for the latents
        logger.info('Posterior mean (actual) - {}'.format(param_name))
        logger.info(latent_loc)
        logger.info('Posterior mean (expected) - {}'.format(param_name))
        logger.info(expected_mean)
        assert_equal(rmse(latent_loc, expected_mean).item(), 0.0, prec=mean_tol)

        # Actual vs expected posterior precisions for the latents
        logger.info('Posterior std (actual) - {}'.format(param_name))
        logger.info(latent_std)
        logger.info('Posterior std (expected) - {}'.format(param_name))
        logger.info(expected_std)
        assert_equal(rmse(latent_std, expected_std).item(), 0.0, prec=std_tol)


def test_logistic_regression():
    dim = 3
    true_coefs = torch.arange(1, dim+1)
    data = torch.randn(2000, dim)
    labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

    def model(data):
        coefs_mean = torch.zeros(dim)
        coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(dim)))
        y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        return y

    nuts_kernel = NUTS(model, step_size=0.0855)
    mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=100).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites='beta')
    assert_equal(rmse(true_coefs, posterior.mean).item(), 0.0, prec=0.1)


def test_bernoulli_beta():
    def model(data):
        alpha = torch.tensor([1.1, 1.1])
        beta = torch.tensor([1.1, 1.1])
        p_latent = pyro.sample("p_latent", dist.Beta(alpha, beta))
        pyro.sample("obs", dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
    nuts_kernel = NUTS(model, step_size=0.02)
    mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=100).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites='p_latent')
    assert_equal(posterior.mean, true_probs, prec=0.02)


def test_normal_gamma():
    def model(data):
        rate = torch.tensor([1.0, 1.0])
        concentration = torch.tensor([1.0, 1.0])
        p_latent = pyro.sample('p_latent', dist.Gamma(rate, concentration))
        pyro.sample("obs", dist.Normal(3, p_latent), obs=data)
        return p_latent

    true_std = torch.tensor([0.5, 2])
    data = dist.Normal(3, true_std).sample(sample_shape=(torch.Size((2000,))))
    nuts_kernel = NUTS(model, step_size=0.01)
    mcmc_run = MCMC(nuts_kernel, num_samples=200, warmup_steps=100).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites='p_latent')
    assert_equal(posterior.mean, true_std, prec=0.05)


def test_logistic_regression_with_dual_averaging():
    dim = 3
    true_coefs = torch.arange(1, dim+1)
    data = torch.randn(2000, dim)
    labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

    def model(data):
        coefs_mean = torch.zeros(dim)
        coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(dim)))
        y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        return y

    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=100).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites='beta')
    assert_equal(rmse(true_coefs, posterior.mean).item(), 0.0, prec=0.1)


@pytest.mark.filterwarnings("ignore:Encountered NAN")
def test_bernoulli_beta_with_dual_averaging():
    def model(data):
        alpha = torch.tensor([1.1, 1.1])
        beta = torch.tensor([1.1, 1.1])
        p_latent = pyro.sample("p_latent", dist.Beta(alpha, beta))
        pyro.sample("obs", dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=100).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites="p_latent")
    assert_equal(posterior.mean, true_probs, prec=0.03)


def test_categorical_dirichlet():
    def model(data):
        concentration = torch.tensor([1.0, 1.0, 1.0])
        p_latent = pyro.sample('p_latent', dist.Dirichlet(concentration))
        pyro.sample("obs", dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(sample_shape=(torch.Size((2000,))))
    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc_run = MCMC(nuts_kernel, num_samples=200, warmup_steps=100).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites='p_latent')
    assert_equal(posterior.mean, true_probs, prec=0.02)
