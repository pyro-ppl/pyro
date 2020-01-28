# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import namedtuple

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.api import MCMC
from tests.common import assert_equal, assert_close

logger = logging.getLogger(__name__)


def mark_jit(*args, **kwargs):
    jit_markers = kwargs.pop("marks", [])
    jit_markers += [
        pytest.mark.skipif('CI' in os.environ,
                           reason='to reduce running time on CI')
    ]
    kwargs["marks"] = jit_markers
    return pytest.param(*args, **kwargs)


def jit_idfn(param):
    return "JIT={}".format(param)


class GaussianChain:

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
        mean_tol=0.08,
        std_tol=0.08,
    ),
    T(
        GaussianChain(dim=10, chain_len=4, num_obs=1),
        num_samples=1600,
        warmup_steps=300,
        hmc_params={'step_size': 0.46,
                    'num_steps': 5},
        expected_means=[0.20, 0.40, 0.60, 0.80],
        expected_precs=[1.25, 0.83, 0.83, 1.25],
        mean_tol=0.08,
        std_tol=0.08,
    ),
    T(
        GaussianChain(dim=5, chain_len=2, num_obs=100),
        num_samples=2000,
        warmup_steps=1000,
        hmc_params={'num_steps': 15, 'step_size': 0.7},
        expected_means=[0.5, 1.0],
        expected_precs=[2.0, 100],
        mean_tol=0.08,
        std_tol=0.08,
    ),
    T(
        GaussianChain(dim=5, chain_len=9, num_obs=1),
        num_samples=3000,
        warmup_steps=500,
        hmc_params={'step_size': 0.2,
                    'num_steps': 15},
        expected_means=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        expected_precs=[1.11, 0.63, 0.48, 0.42, 0.4, 0.42, 0.48, 0.63, 1.11],
        mean_tol=0.11,
        std_tol=0.11,
    )
]

TEST_IDS = [t[0].id_fn() if type(t).__name__ == 'TestExample'
            else t[0][0].id_fn() for t in TEST_CASES]


@pytest.mark.parametrize(
    'fixture, num_samples, warmup_steps, hmc_params, expected_means, expected_precs, mean_tol, std_tol',
    TEST_CASES,
    ids=TEST_IDS)
@pytest.mark.skip(reason='Slow test (https://github.com/pytorch/pytorch/issues/12190)')
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
    hmc_kernel = HMC(fixture.model, **hmc_params)
    samples = MCMC(hmc_kernel, num_samples, warmup_steps).run(fixture.data)
    for i in range(1, fixture.chain_len + 1):
        param_name = 'loc_' + str(i)
        marginal = samples[param_name]
        latent_loc = marginal.mean(0)
        latent_std = marginal.var(0).sqrt()
        expected_mean = torch.ones(fixture.dim) * expected_means[i - 1]
        expected_std = 1 / torch.sqrt(torch.ones(fixture.dim) * expected_precs[i - 1])

        # Actual vs expected posterior means for the latents
        logger.debug('Posterior mean (actual) - {}'.format(param_name))
        logger.debug(latent_loc)
        logger.debug('Posterior mean (expected) - {}'.format(param_name))
        logger.debug(expected_mean)
        assert_equal(rmse(latent_loc, expected_mean).item(), 0.0, prec=mean_tol)

        # Actual vs expected posterior precisions for the latents
        logger.debug('Posterior std (actual) - {}'.format(param_name))
        logger.debug(latent_std)
        logger.debug('Posterior std (expected) - {}'.format(param_name))
        logger.debug(expected_std)
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
        coefs_mean = pyro.param('coefs_mean', torch.zeros(dim))
        coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(dim)))
        y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        return y

    hmc_kernel = HMC(model, step_size=step_size, trajectory_length=trajectory_length,
                     num_steps=num_steps, adapt_step_size=adapt_step_size,
                     adapt_mass_matrix=adapt_mass_matrix, full_mass=full_mass)
    mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100, disable_progbar=True)
    mcmc.run(data)
    samples = mcmc.get_samples()['beta']
    assert_equal(rmse(true_coefs, samples.mean(0)).item(), 0.0, prec=0.1)


@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
def test_dirichlet_categorical(jit):
    def model(data):
        concentration = torch.tensor([1.0, 1.0, 1.0])
        p_latent = pyro.sample('p_latent', dist.Dirichlet(concentration))
        pyro.sample("obs", dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(sample_shape=(torch.Size((2000,))))
    hmc_kernel = HMC(model, trajectory_length=1, jit_compile=jit, ignore_jit_warnings=True)
    mcmc = MCMC(hmc_kernel, num_samples=200, warmup_steps=100)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(samples['p_latent'].mean(0), true_probs, prec=0.02)


@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
def test_beta_bernoulli(jit):
    def model(data):
        alpha = torch.tensor([1.1, 1.1])
        beta = torch.tensor([1.1, 1.1])
        p_latent = pyro.sample('p_latent', dist.Beta(alpha, beta))
        with pyro.plate("data", data.shape[0], dim=-2):
            pyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
    hmc_kernel = HMC(model, trajectory_length=1, max_plate_nesting=2,
                     jit_compile=jit, ignore_jit_warnings=True)
    mcmc = MCMC(hmc_kernel, num_samples=800, warmup_steps=500)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(samples['p_latent'].mean(0), true_probs, prec=0.05)


def test_gamma_normal():
    def model(data):
        rate = torch.tensor([1.0, 1.0])
        concentration = torch.tensor([1.0, 1.0])
        p_latent = pyro.sample('p_latent', dist.Gamma(rate, concentration))
        pyro.sample("obs", dist.Normal(3, p_latent), obs=data)
        return p_latent

    true_std = torch.tensor([0.5, 2])
    data = dist.Normal(3, true_std).sample(sample_shape=(torch.Size((2000,))))
    hmc_kernel = HMC(model, num_steps=15, step_size=0.01, adapt_step_size=True)
    mcmc = MCMC(hmc_kernel, num_samples=200, warmup_steps=200)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(samples['p_latent'].mean(0), true_std, prec=0.05)


@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
def test_bernoulli_latent_model(jit):
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
                     jit_compile=jit, ignore_jit_warnings=True)
    mcmc = MCMC(hmc_kernel, num_samples=600, warmup_steps=200)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(samples['y_prob'].mean(0), y_prob, prec=0.06)


@pytest.mark.parametrize("kernel", [HMC, NUTS])
@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
@pytest.mark.skipif("CUDA_TEST" in os.environ, reason="https://github.com/pytorch/pytorch/issues/22811")
def test_unnormalized_normal(kernel, jit):
    true_mean, true_std = torch.tensor(5.), torch.tensor(1.)
    init_params = {"z": torch.tensor(0.)}

    def potential_energy(params):
        return 0.5 * torch.sum(((params["z"] - true_mean) / true_std) ** 2)

    potential_fn = potential_energy if not jit else torch.jit.trace(potential_energy, init_params)
    hmc_kernel = kernel(model=None, potential_fn=potential_fn)

    samples = init_params
    warmup_steps = 400
    hmc_kernel.initial_params = samples
    hmc_kernel.setup(warmup_steps)

    for i in range(warmup_steps):
        samples = hmc_kernel(samples)

    posterior = []
    for i in range(2000):
        hmc_kernel.clear_cache()
        samples = hmc_kernel(samples)
        posterior.append(samples)

    posterior = torch.stack([sample["z"] for sample in posterior])
    assert_close(torch.mean(posterior), true_mean, rtol=0.05)
    assert_close(torch.std(posterior), true_std, rtol=0.05)


@pytest.mark.parametrize('jit', [False, mark_jit(True)], ids=jit_idfn)
@pytest.mark.parametrize('op', [torch.inverse, torch.cholesky])
def test_singular_matrix_catch(jit, op):
    def potential_energy(z):
        return op(z['cov']).sum()

    init_params = {'cov': torch.eye(3)}
    potential_fn = potential_energy if not jit else torch.jit.trace(potential_energy, init_params)
    hmc_kernel = HMC(potential_fn=potential_fn, adapt_step_size=False,
                     num_steps=10, step_size=1e-20)
    hmc_kernel.initial_params = init_params
    hmc_kernel.setup(warmup_steps=0)
    # setup an invalid cache to trigger singular error for torch.inverse
    hmc_kernel._cache({'cov': torch.ones(3, 3)}, torch.tensor(0.), {'cov': torch.zeros(3, 3)})

    samples = init_params
    for i in range(10):
        samples = hmc_kernel.sample(samples)
