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
import pyro.poutine as poutine
from tests.common import assert_equal

from .test_hmc import TEST_CASES, TEST_IDS, T, rmse

logger = logging.getLogger(__name__)

T2 = T(*TEST_CASES[2].values)._replace(num_samples=800, warmup_steps=200)
TEST_CASES[2] = pytest.param(*T2, marks=pytest.mark.skipif(
    'CI' in os.environ and os.environ['CI'] == 'true', reason='Slow test - skip on CI'))
T3 = T(*TEST_CASES[3].values)._replace(num_samples=1000, warmup_steps=200)
TEST_CASES[3] = pytest.param(*T3, marks=[
    pytest.mark.skipif('CI' in os.environ and os.environ['CI'] == 'true',
                       reason='Slow test - skip on CI')]
)


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
    data = torch.randn(2000, dim)
    true_coefs = torch.arange(1., dim + 1.)
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


def test_beta_bernoulli():
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


def test_gamma_normal():
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
    data = torch.randn(2000, dim)
    true_coefs = torch.arange(1., dim + 1.)
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


def test_beta_bernoulli_with_dual_averaging():
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


def test_dirichlet_categorical():
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


def test_gamma_beta():
    def model(data):
        alpha_prior = pyro.sample('alpha', dist.Gamma(concentration=1., rate=1.))
        beta_prior = pyro.sample('beta', dist.Gamma(concentration=1., rate=1.))
        pyro.sample('x', dist.Beta(concentration1=alpha_prior, concentration0=beta_prior), obs=data)

    true_alpha = torch.tensor(5.)
    true_beta = torch.tensor(1.)
    data = dist.Beta(concentration1=true_alpha, concentration0=true_beta).sample(torch.Size((5000,)))
    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=200).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites=['alpha', 'beta'])
    assert_equal(posterior.mean, torch.stack([true_alpha, true_beta]), prec=0.05)


def test_gaussian_mixture_model():
    K, N = 3, 1000

    @poutine.broadcast
    def gmm(data):
        with pyro.iarange("num_clusters", K):
            mix_proportions = pyro.sample("phi", dist.Dirichlet(torch.tensor(1.)))
            cluster_means = pyro.sample("cluster_means", dist.Normal(torch.arange(float(K)), 1.))
        with pyro.iarange("data", data.shape[0]):
            assignments = pyro.sample("assignments", dist.Categorical(mix_proportions))
            pyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)
        return cluster_means

    true_cluster_means = torch.tensor([1., 5., 10.])
    true_mix_proportions = torch.tensor([0.1, 0.3, 0.6])
    cluster_assignments = dist.Categorical(true_mix_proportions).sample(torch.Size((N,)))
    data = dist.Normal(true_cluster_means[cluster_assignments], 1.0).sample()
    nuts_kernel = NUTS(gmm, adapt_step_size=True, max_iarange_nesting=1)
    mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=200).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites=["phi", "cluster_means"]).mean.sort()[0]
    assert_equal(posterior[0], true_mix_proportions, prec=0.05)
    assert_equal(posterior[1], true_cluster_means, prec=0.2)


def test_bernoulli_latent_model():
    @poutine.broadcast
    def model(data):
        y_prob = pyro.sample("y_prob", dist.Beta(1., 1.))
        with pyro.iarange("data", data.shape[0]):
            y = pyro.sample("y", dist.Bernoulli(y_prob))
            z = pyro.sample("z", dist.Bernoulli(0.65 * y + 0.1))
            pyro.sample("obs", dist.Normal(2. * z, 1.), obs=data)

    N = 2000
    y_prob = torch.tensor(0.3)
    y = dist.Bernoulli(y_prob).sample(torch.Size((N,)))
    z = dist.Bernoulli(0.65 * y + 0.1).sample()
    data = dist.Normal(2. * z, 1.0).sample()
    nuts_kernel = NUTS(model, adapt_step_size=True, max_iarange_nesting=1)
    mcmc_run = MCMC(nuts_kernel, num_samples=600, warmup_steps=200).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites="y_prob").mean
    assert_equal(posterior, y_prob, prec=0.05)


@pytest.mark.parametrize("num_steps,use_einsum", [
    (2, False),
    (3, False),
    # This will crash without the einsum backend
    pytest.param(30, True,
                 marks=pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/10661")),
])
def test_gaussian_hmm_enum_shape(num_steps, use_einsum):
    dim = 4

    def model(data):
        initialize = pyro.sample("initialize", dist.Dirichlet(torch.ones(dim)))
        transition = pyro.sample("transition", dist.Dirichlet(torch.ones(dim, dim)))
        emission_loc = pyro.sample("emission_loc", dist.Normal(torch.zeros(dim), torch.ones(dim)))
        emission_scale = pyro.sample("emission_scale", dist.LogNormal(torch.zeros(dim), torch.ones(dim)))
        x = None
        for t, y in enumerate(data):
            x = pyro.sample("x_{}".format(t), dist.Categorical(initialize if x is None else transition[x]))
            pyro.sample("y_{}".format(t), dist.Normal(emission_loc[x], emission_scale[x]), obs=y)
            # check shape
            effective_dim = sum(1 for size in x.shape if size > 1)
            assert effective_dim == 1

    data = torch.ones(num_steps)
    nuts_kernel = NUTS(model, adapt_step_size=True, max_iarange_nesting=0,
                       experimental_use_einsum=use_einsum)
    MCMC(nuts_kernel, num_samples=5, warmup_steps=5).run(data)
