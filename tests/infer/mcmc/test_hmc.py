from __future__ import absolute_import, division, print_function

import logging
import os
from collections import namedtuple

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC
from tests.common import assert_equal

logger = logging.getLogger(__name__)


class GaussianChain(object):

    def __init__(self, dim, chain_len, num_obs):
        self.dim = dim
        self.chain_len = chain_len
        self.num_obs = num_obs
        self.loc_0 = torch.zeros(self.dim)
        self.lambda_prec = torch.ones(self.dim)

    def model(self, data):
        loc = self.loc_0
        lambda_prec = self.lambda_prec
        for i in range(1, self.chain_len + 1):
            loc = pyro.sample('loc_{}'.format(i),
                              dist.Normal(loc=loc, scale=lambda_prec))
        pyro.sample('obs', dist.Normal(loc, lambda_prec), obs=data)

    @property
    def data(self):
        return torch.ones(self.num_obs, self.dim)

    def id_fn(self):
        return 'dim={}_chain-len={}_num_obs={}'.format(self.dim, self.chain_len, self.num_obs)


def rmse(t1, t2):
    return (t1 - t2).pow(2).mean().sqrt()


T = namedtuple('TestExample', [
    'fixture',
    'num_samples',
    'warmup_steps',
    'hmc_params',
    'expected_means',
    'expected_precs',
    'mean_tol',
    'std_tol'])

TEST_CASES = [
    T(
        GaussianChain(dim=10, chain_len=3, num_obs=1),
        num_samples=800,
        warmup_steps=200,
        hmc_params={'step_size': 0.5,
                    'num_steps': 4},
        expected_means=[0.25, 0.50, 0.75],
        expected_precs=[1.33, 1, 1.33],
        mean_tol=0.06,
        std_tol=0.08,
    ),
    T(
        GaussianChain(dim=10, chain_len=4, num_obs=1),
        num_samples=1200,
        warmup_steps=300,
        hmc_params={'step_size': 0.46,
                    'num_steps': 5},
        expected_means=[0.20, 0.40, 0.60, 0.80],
        expected_precs=[1.25, 0.83, 0.83, 1.25],
        mean_tol=0.06,
        std_tol=0.06,
    ),
    # XXX: Very sensitive to HMC parameters. Biased estimate is obtained
    # without enough samples and/or larger step size.
    pytest.param(*T(
        GaussianChain(dim=5, chain_len=2, num_obs=10000),
        num_samples=2000,
        warmup_steps=500,
        hmc_params={'step_size': 0.013,
                    'num_steps': 25},
        expected_means=[0.5, 1.0],
        expected_precs=[2.0, 10000],
        mean_tol=0.05,
        std_tol=0.05,
    ), marks=[pytest.mark.xfail(reason="flaky"),
              pytest.mark.skipif('CI' in os.environ or 'CUDA_TEST' in os.environ,
                                 reason='Slow test - skip on CI/CUDA')]),
    pytest.param(*T(
        GaussianChain(dim=5, chain_len=9, num_obs=1),
        num_samples=3000,
        warmup_steps=500,
        hmc_params={'step_size': 0.3,
                    'num_steps': 8},
        expected_means=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        expected_precs=[1.11, 0.63, 0.48, 0.42, 0.4, 0.42, 0.48, 0.63, 1.11],
        mean_tol=0.08,
        std_tol=0.08,
    ), marks=[pytest.mark.xfail(reason="flaky"),
              pytest.mark.skipif('CI' in os.environ or 'CUDA_TEST' in os.environ,
                                 reason='Slow test - skip on CI/CUDA')])
]

TEST_IDS = [t[0].id_fn() if type(t).__name__ == 'TestExample'
            else t[0][0].id_fn() for t in TEST_CASES]


@pytest.mark.parametrize(
    'fixture, num_samples, warmup_steps, hmc_params, expected_means, expected_precs, mean_tol, std_tol',
    TEST_CASES,
    ids=TEST_IDS)
@pytest.mark.init(rng_seed=34)
@pytest.mark.disable_validation()
def test_hmc_conjugate_gaussian(fixture,
                                num_samples,
                                warmup_steps,
                                hmc_params,
                                expected_means,
                                expected_precs,
                                mean_tol,
                                std_tol):
    pyro.get_param_store().clear()
    hmc_params["adapt_step_size"] = False
    hmc_params["adapt_mass_matrix"] = False
    hmc_kernel = HMC(fixture.model, **hmc_params)
    mcmc_run = MCMC(hmc_kernel, num_samples, warmup_steps).run(fixture.data)
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


@pytest.mark.parametrize(
    "step_size, trajectory_length, num_steps, adapt_step_size, adapt_mass_matrix, full_mass",
    [
        (0.0855, None, 4, False, False, False),
        (0.0855, None, 4, False, True, False),
        (None, 1, None, True, False, False),
        (None, 1, None, True, True, False),
        (None, 1, None, True, True, True),
    ]
)
def test_logistic_regression(step_size, trajectory_length, num_steps,
                             adapt_step_size, adapt_mass_matrix, full_mass):
    dim = 3
    data = torch.randn(2000, dim)
    true_coefs = torch.arange(1., dim + 1.)
    labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

    def model(data):
        coefs_mean = torch.zeros(dim)
        coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(dim)))
        y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        return y

    hmc_kernel = HMC(model, step_size, trajectory_length, num_steps,
                     adapt_step_size, adapt_mass_matrix, full_mass)
    mcmc_run = MCMC(hmc_kernel, num_samples=500, warmup_steps=100).run(data)
    beta_posterior = EmpiricalMarginal(mcmc_run, sites='beta')
    assert_equal(rmse(true_coefs, beta_posterior.mean).item(), 0.0, prec=0.1)


def test_beta_bernoulli():
    def model(data):
        # wrapped by `pyro.param` to test if it works
        alpha = pyro.param('alpha', torch.tensor([1.1, 1.1]))
        beta = pyro.param('beta', torch.tensor([1.1, 1.1]))
        p_latent = pyro.sample('p_latent', dist.Beta(alpha, beta))
        pyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
    hmc_kernel = HMC(model, trajectory_length=1)
    mcmc_run = MCMC(hmc_kernel, num_samples=800, warmup_steps=500).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites='p_latent')
    assert_equal(posterior.mean, true_probs, prec=0.05)


def test_gamma_normal():
    def model(data):
        rate = torch.tensor([1.0, 1.0])
        concentration = torch.tensor([1.0, 1.0])
        p_latent = pyro.sample('p_latent', dist.Gamma(rate, concentration))
        pyro.sample("obs", dist.Normal(3, p_latent), obs=data)
        return p_latent

    true_std = torch.tensor([0.5, 2])
    data = dist.Normal(3, true_std).sample(sample_shape=(torch.Size((2000,))))
    hmc_kernel = HMC(model, trajectory_length=1)
    mcmc_run = MCMC(hmc_kernel, num_samples=500, warmup_steps=100).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites='p_latent')
    assert_equal(posterior.mean, true_std, prec=0.05)


def test_dirichlet_categorical():
    def model(data):
        concentration = torch.tensor([1.0, 1.0, 1.0])
        p_latent = pyro.sample('p_latent', dist.Dirichlet(concentration))
        pyro.sample("obs", dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(sample_shape=(torch.Size((2000,))))
    hmc_kernel = HMC(model, trajectory_length=1)
    mcmc_run = MCMC(hmc_kernel, num_samples=200, warmup_steps=100).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites='p_latent')
    assert_equal(posterior.mean, true_probs, prec=0.02)


def test_gaussian_mixture_model():
    K, N = 3, 1000

    def gmm(data):
        mix_proportions = pyro.sample("phi", dist.Dirichlet(torch.ones(K)))
        with pyro.plate("num_clusters", K):
            cluster_means = pyro.sample("cluster_means", dist.Normal(torch.arange(float(K)), 1.))
        with pyro.plate("data", data.shape[0]):
            assignments = pyro.sample("assignments", dist.Categorical(mix_proportions))
            pyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)
        return cluster_means

    true_cluster_means = torch.tensor([1., 5., 10.])
    true_mix_proportions = torch.tensor([0.1, 0.3, 0.6])
    cluster_assignments = dist.Categorical(true_mix_proportions).sample(torch.Size((N,)))
    data = dist.Normal(true_cluster_means[cluster_assignments], 1.0).sample()
    hmc_kernel = HMC(gmm, trajectory_length=1, max_plate_nesting=1)
    mcmc_run = MCMC(hmc_kernel, num_samples=300, warmup_steps=100).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites=["phi", "cluster_means"]).mean.sort()[0]
    assert_equal(posterior[0], true_mix_proportions, prec=0.05)
    assert_equal(posterior[1], true_cluster_means, prec=0.2)


@pytest.mark.parametrize("use_einsum", [False, True])
def test_bernoulli_latent_model(use_einsum):
    def model(data):
        y_prob = pyro.sample("y_prob", dist.Beta(1.0, 1.0))
        y = pyro.sample("y", dist.Bernoulli(y_prob))
        with pyro.plate("data", data.shape[0]):
            z = pyro.sample("z", dist.Bernoulli(0.65 * y + 0.1))
            pyro.sample("obs", dist.Normal(2. * z, 1.), obs=data)
        pyro.sample("nuisance", dist.Bernoulli(0.3))

    N = 2000
    y_prob = torch.tensor(0.3)
    y = dist.Bernoulli(y_prob).sample(torch.Size((N,)))
    z = dist.Bernoulli(0.65 * y + 0.1).sample()
    data = dist.Normal(2. * z, 1.0).sample()
    hmc_kernel = HMC(model, trajectory_length=1, max_plate_nesting=1,
                     experimental_use_einsum=use_einsum)
    mcmc_run = MCMC(hmc_kernel, num_samples=600, warmup_steps=200).run(data)
    posterior = EmpiricalMarginal(mcmc_run, sites="y_prob").mean
    assert_equal(posterior, y_prob, prec=0.05)
