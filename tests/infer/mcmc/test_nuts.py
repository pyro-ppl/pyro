# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import namedtuple

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
from pyro.contrib.conjugate.infer import BetaBinomialPair, collapse_conjugate, GammaPoissonPair, posterior_replay
from pyro.infer import TraceEnum_ELBO, SVI
from pyro.infer.mcmc import ArrowheadMassMatrix, MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.util import ignore_jit_warnings
from tests.common import assert_close, assert_equal

from .test_hmc import GaussianChain, rmse

logger = logging.getLogger(__name__)


T = namedtuple('TestExample', [
    'fixture',
    'num_samples',
    'warmup_steps',
    'expected_means',
    'expected_precs',
    'mean_tol',
    'std_tol'])

TEST_CASES = [
    T(
        GaussianChain(dim=10, chain_len=3, num_obs=1),
        num_samples=800,
        warmup_steps=200,
        expected_means=[0.25, 0.50, 0.75],
        expected_precs=[1.33, 1, 1.33],
        mean_tol=0.09,
        std_tol=0.09,
    ),
    T(
        GaussianChain(dim=10, chain_len=4, num_obs=1),
        num_samples=1600,
        warmup_steps=200,
        expected_means=[0.20, 0.40, 0.60, 0.80],
        expected_precs=[1.25, 0.83, 0.83, 1.25],
        mean_tol=0.07,
        std_tol=0.06,
    ),
    T(
        GaussianChain(dim=5, chain_len=2, num_obs=10000),
        num_samples=800,
        warmup_steps=200,
        expected_means=[0.5, 1.0],
        expected_precs=[2.0, 10000],
        mean_tol=0.05,
        std_tol=0.05,
    ),
    T(
        GaussianChain(dim=5, chain_len=9, num_obs=1),
        num_samples=1400,
        warmup_steps=200,
        expected_means=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        expected_precs=[1.11, 0.63, 0.48, 0.42, 0.4, 0.42, 0.48, 0.63, 1.11],
        mean_tol=0.08,
        std_tol=0.08,
    )
]


TEST_IDS = [t[0].id_fn() if type(t).__name__ == 'TestExample'
            else t[0][0].id_fn() for t in TEST_CASES]


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


@pytest.mark.parametrize(
    'fixture, num_samples, warmup_steps, expected_means, expected_precs, mean_tol, std_tol',
    TEST_CASES,
    ids=TEST_IDS)
@pytest.mark.skip(reason='Slow test (https://github.com/pytorch/pytorch/issues/12190)')
@pytest.mark.disable_validation()
def test_nuts_conjugate_gaussian(fixture,
                                 num_samples,
                                 warmup_steps,
                                 expected_means,
                                 expected_precs,
                                 mean_tol,
                                 std_tol):
    pyro.get_param_store().clear()
    nuts_kernel = NUTS(fixture.model)
    mcmc = MCMC(nuts_kernel, num_samples, warmup_steps)
    mcmc.run(fixture.data)
    samples = mcmc.get_samples()
    for i in range(1, fixture.chain_len + 1):
        param_name = 'loc_' + str(i)
        latent = samples[param_name]
        latent_loc = latent.mean(0)
        latent_std = latent.std(0)
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


@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
@pytest.mark.parametrize("use_multinomial_sampling", [True, False])
def test_logistic_regression(jit, use_multinomial_sampling):
    dim = 3
    data = torch.randn(2000, dim)
    true_coefs = torch.arange(1., dim + 1.)
    labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

    def model(data):
        coefs_mean = torch.zeros(dim)
        coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(dim)))
        y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        return y

    nuts_kernel = NUTS(model,
                       use_multinomial_sampling=use_multinomial_sampling,
                       jit_compile=jit,
                       ignore_jit_warnings=True)
    mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(rmse(true_coefs, samples["beta"].mean(0)).item(), 0.0, prec=0.1)


@pytest.mark.parametrize(
    "step_size, adapt_step_size, adapt_mass_matrix, full_mass",
    [
        (0.1, False, False, False),
        (0.5, False, True, False),
        (None, True, False, False),
        (None, True, True, False),
        (None, True, True, True),
    ]
)
def test_beta_bernoulli(step_size, adapt_step_size, adapt_mass_matrix, full_mass):
    def model(data):
        alpha = torch.tensor([1.1, 1.1])
        beta = torch.tensor([1.1, 1.1])
        p_latent = pyro.sample("p_latent", dist.Beta(alpha, beta))
        pyro.sample("obs", dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
    nuts_kernel = NUTS(model, step_size=step_size, adapt_step_size=adapt_step_size,
                       adapt_mass_matrix=adapt_mass_matrix, full_mass=full_mass)
    mcmc = MCMC(nuts_kernel, num_samples=400, warmup_steps=200)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(samples["p_latent"].mean(0), true_probs, prec=0.02)


@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
@pytest.mark.parametrize("use_multinomial_sampling", [True, False])
def test_gamma_normal(jit, use_multinomial_sampling):
    def model(data):
        rate = torch.tensor([1.0, 1.0])
        concentration = torch.tensor([1.0, 1.0])
        p_latent = pyro.sample('p_latent', dist.Gamma(rate, concentration))
        pyro.sample("obs", dist.Normal(3, p_latent), obs=data)
        return p_latent

    true_std = torch.tensor([0.5, 2])
    data = dist.Normal(3, true_std).sample(sample_shape=(torch.Size((2000,))))
    nuts_kernel = NUTS(model,
                       use_multinomial_sampling=use_multinomial_sampling,
                       jit_compile=jit,
                       ignore_jit_warnings=True)
    mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=100)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(samples["p_latent"].mean(0), true_std, prec=0.05)


@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
def test_dirichlet_categorical(jit):
    def model(data):
        concentration = torch.tensor([1.0, 1.0, 1.0])
        p_latent = pyro.sample('p_latent', dist.Dirichlet(concentration))
        pyro.sample("obs", dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(sample_shape=(torch.Size((2000,))))
    nuts_kernel = NUTS(model, jit_compile=jit, ignore_jit_warnings=True)
    mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=100)
    mcmc.run(data)
    samples = mcmc.get_samples()
    posterior = samples["p_latent"]
    assert_equal(posterior.mean(0), true_probs, prec=0.02)


@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
def test_gamma_beta(jit):
    def model(data):
        alpha_prior = pyro.sample('alpha', dist.Gamma(concentration=1., rate=1.))
        beta_prior = pyro.sample('beta', dist.Gamma(concentration=1., rate=1.))
        pyro.sample('x', dist.Beta(concentration1=alpha_prior, concentration0=beta_prior), obs=data)

    true_alpha = torch.tensor(5.)
    true_beta = torch.tensor(1.)
    data = dist.Beta(concentration1=true_alpha, concentration0=true_beta).sample(torch.Size((5000,)))
    nuts_kernel = NUTS(model, jit_compile=jit, ignore_jit_warnings=True)
    mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(samples["alpha"].mean(0), true_alpha, prec=0.08)
    assert_equal(samples["beta"].mean(0), true_beta, prec=0.05)


@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
def test_gaussian_mixture_model(jit):
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
    nuts_kernel = NUTS(gmm, max_plate_nesting=1, jit_compile=jit, ignore_jit_warnings=True)
    mcmc = MCMC(nuts_kernel, num_samples=300, warmup_steps=100)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(samples["phi"].mean(0).sort()[0], true_mix_proportions, prec=0.05)
    assert_equal(samples["cluster_means"].mean(0).sort()[0], true_cluster_means, prec=0.2)


@pytest.mark.parametrize("jit", [False, mark_jit(True)], ids=jit_idfn)
def test_bernoulli_latent_model(jit):
    @poutine.broadcast
    def model(data):
        y_prob = pyro.sample("y_prob", dist.Beta(1., 1.))
        with pyro.plate("data", data.shape[0]):
            y = pyro.sample("y", dist.Bernoulli(y_prob))
            z = pyro.sample("z", dist.Bernoulli(0.65 * y + 0.1))
            pyro.sample("obs", dist.Normal(2. * z, 1.), obs=data)

    N = 2000
    y_prob = torch.tensor(0.3)
    y = dist.Bernoulli(y_prob).sample(torch.Size((N,)))
    z = dist.Bernoulli(0.65 * y + 0.1).sample()
    data = dist.Normal(2. * z, 1.0).sample()
    nuts_kernel = NUTS(model, max_plate_nesting=1, jit_compile=jit, ignore_jit_warnings=True)
    mcmc = MCMC(nuts_kernel, num_samples=600, warmup_steps=200)
    mcmc.run(data)
    samples = mcmc.get_samples()
    assert_equal(samples["y_prob"].mean(0), y_prob, prec=0.05)


@pytest.mark.parametrize("num_steps", [2, 3, 30])
def test_gaussian_hmm(num_steps):
    dim = 4

    def model(data):
        initialize = pyro.sample("initialize", dist.Dirichlet(torch.ones(dim)))
        with pyro.plate("states", dim):
            transition = pyro.sample("transition", dist.Dirichlet(torch.ones(dim, dim)))
            emission_loc = pyro.sample("emission_loc", dist.Normal(torch.zeros(dim), torch.ones(dim)))
            emission_scale = pyro.sample("emission_scale", dist.LogNormal(torch.zeros(dim), torch.ones(dim)))
        x = None
        with ignore_jit_warnings([("Iterating over a tensor", RuntimeWarning)]):
            for t, y in pyro.markov(enumerate(data)):
                x = pyro.sample("x_{}".format(t),
                                dist.Categorical(initialize if x is None else transition[x]),
                                infer={"enumerate": "parallel"})
                pyro.sample("y_{}".format(t), dist.Normal(emission_loc[x], emission_scale[x]), obs=y)

    def _get_initial_trace():
        guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: not msg["name"].startswith("x") and
                                        not msg["name"].startswith("y")))
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        svi = SVI(model, guide, optim.Adam({"lr": .01}), elbo)
        for _ in range(100):
            svi.step(data)
        return poutine.trace(guide).get_trace(data)

    def _generate_data():
        transition_probs = torch.rand(dim, dim)
        emissions_loc = torch.arange(dim, dtype=torch.Tensor().dtype)
        emissions_scale = 1.
        state = torch.tensor(1)
        obs = [dist.Normal(emissions_loc[state], emissions_scale).sample()]
        for _ in range(num_steps):
            state = dist.Categorical(transition_probs[state]).sample()
            obs.append(dist.Normal(emissions_loc[state], emissions_scale).sample())
        return torch.stack(obs)

    data = _generate_data()
    nuts_kernel = NUTS(model, max_plate_nesting=1, jit_compile=True, ignore_jit_warnings=True)
    if num_steps == 30:
        nuts_kernel.initial_trace = _get_initial_trace()
    mcmc = MCMC(nuts_kernel, num_samples=5, warmup_steps=5)
    mcmc.run(data)


@pytest.mark.parametrize("hyperpriors", [False, True])
def test_beta_binomial(hyperpriors):
    def model(data):
        with pyro.plate("plate_0", data.shape[-1]):
            alpha = pyro.sample("alpha", dist.HalfCauchy(1.)) if hyperpriors else torch.tensor([1., 1.])
            beta = pyro.sample("beta", dist.HalfCauchy(1.)) if hyperpriors else torch.tensor([1., 1.])
            beta_binom = BetaBinomialPair()
            with pyro.plate("plate_1", data.shape[-2]):
                probs = pyro.sample("probs", beta_binom.latent(alpha, beta))
                with pyro.plate("data", data.shape[0]):
                    pyro.sample("binomial", beta_binom.conditional(probs=probs, total_count=total_count), obs=data)

    true_probs = torch.tensor([[0.7, 0.4], [0.6, 0.4]])
    total_count = torch.tensor([[1000, 600], [400, 800]])
    num_samples = 80
    data = dist.Binomial(total_count=total_count, probs=true_probs).sample(sample_shape=(torch.Size((10,))))
    hmc_kernel = NUTS(collapse_conjugate(model), jit_compile=True, ignore_jit_warnings=True)
    mcmc = MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=50)
    mcmc.run(data)
    samples = mcmc.get_samples()
    posterior = posterior_replay(model, samples, data, num_samples=num_samples)
    assert_equal(posterior["probs"].mean(0), true_probs, prec=0.05)


@pytest.mark.parametrize("hyperpriors", [False, True])
def test_gamma_poisson(hyperpriors):
    def model(data):
        with pyro.plate("latent_dim", data.shape[1]):
            alpha = pyro.sample("alpha", dist.HalfCauchy(1.)) if hyperpriors else torch.tensor([1., 1.])
            beta = pyro.sample("beta", dist.HalfCauchy(1.)) if hyperpriors else torch.tensor([1., 1.])
            gamma_poisson = GammaPoissonPair()
            rate = pyro.sample("rate", gamma_poisson.latent(alpha, beta))
            with pyro.plate("data", data.shape[0]):
                pyro.sample("obs", gamma_poisson.conditional(rate), obs=data)

    true_rate = torch.tensor([3., 10.])
    num_samples = 100
    data = dist.Poisson(rate=true_rate).sample(sample_shape=(torch.Size((100,))))
    hmc_kernel = NUTS(collapse_conjugate(model), jit_compile=True, ignore_jit_warnings=True)
    mcmc = MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=50)
    mcmc.run(data)
    samples = mcmc.get_samples()
    posterior = posterior_replay(model, samples, data, num_samples=num_samples)
    assert_equal(posterior["rate"].mean(0), true_rate, prec=0.3)


def test_structured_mass():
    def model(cov):
        w = pyro.sample("w", dist.Normal(0, 1000).expand([2]).to_event(1))
        x = pyro.sample("x", dist.Normal(0, 1000).expand([1]).to_event(1))
        y = pyro.sample("y", dist.Normal(0, 1000).expand([1]).to_event(1))
        z = pyro.sample("z", dist.Normal(0, 1000).expand([1]).to_event(1))
        wxyz = torch.cat([w, x, y, z])
        pyro.sample("obs", dist.MultivariateNormal(torch.zeros(5), cov), obs=wxyz)

    w_cov = torch.tensor([[1.5, 0.5], [0.5, 1.5]])
    xy_cov = torch.tensor([[2., 1.], [1., 3.]])
    z_var = torch.tensor([2.5])
    cov = torch.zeros(5, 5)
    cov[:2, :2] = w_cov
    cov[2:4, 2:4] = xy_cov
    cov[4, 4] = z_var

    # smoke tests
    for dense_mass in [True, False]:
        kernel = NUTS(model, jit_compile=True, ignore_jit_warnings=True, full_mass=dense_mass)
        mcmc = MCMC(kernel, num_samples=1, warmup_steps=1)
        mcmc.run(cov)
        assert kernel.inverse_mass_matrix[("w", "x", "y", "z")].dim() == 1 + int(dense_mass)

    kernel = NUTS(model, jit_compile=True, ignore_jit_warnings=True, full_mass=[("w",), ("x", "y")])
    mcmc = MCMC(kernel, num_samples=1, warmup_steps=1000)
    mcmc.run(cov)
    assert_close(kernel.inverse_mass_matrix[("w",)], w_cov, atol=0.5, rtol=0.5)
    assert_close(kernel.inverse_mass_matrix[("x", "y")], xy_cov, atol=0.5, rtol=0.5)
    assert_close(kernel.inverse_mass_matrix[("z",)], z_var, atol=0.5, rtol=0.5)


def test_arrowhead_mass():
    def model(prec):
        w = pyro.sample("w", dist.Normal(0, 1000).expand([2]).to_event(1))
        x = pyro.sample("x", dist.Normal(0, 1000).expand([1]).to_event(1))
        y = pyro.sample("y", dist.Normal(0, 1000).expand([1]).to_event(1))
        z = pyro.sample("z", dist.Normal(0, 1000).expand([2]).to_event(1))
        wyxz = torch.cat([w, y, x, z])
        pyro.sample("obs", dist.MultivariateNormal(torch.zeros(6), precision_matrix=prec), obs=wyxz)

    A = torch.randn(6, 12)
    prec = A @ A.t() * 0.1

    # smoke tests
    for dense_mass in [True, False]:
        kernel = NUTS(model, jit_compile=True, ignore_jit_warnings=True, full_mass=dense_mass)
        mcmc = MCMC(kernel, num_samples=1, warmup_steps=1)
        mcmc.run(prec)
        assert kernel.inverse_mass_matrix[("w", "x", "y", "z")].dim() == 1 + int(dense_mass)

    kernel = NUTS(model, jit_compile=True, ignore_jit_warnings=True, full_mass=[("w",), ("y", "x")])
    kernel.mass_matrix_adapter = ArrowheadMassMatrix()
    mcmc = MCMC(kernel, num_samples=1, warmup_steps=1000)
    mcmc.run(prec)
    assert ("w", "y", "x", "z") in kernel.inverse_mass_matrix
    mass_matrix = kernel.mass_matrix_adapter.mass_matrix[("w", "y", "x", "z")]
    assert mass_matrix.top.shape == (4, 6)
    assert mass_matrix.bottom_diag.shape == (2,)
    assert_close(mass_matrix.top, prec[:4], atol=0.2, rtol=0.2)
    assert_close(mass_matrix.bottom_diag, prec.diag()[4:], atol=0.2, rtol=0.2)


def test_dirichlet_categorical_grad_adapt():
    def model(data):
        concentration = torch.tensor([1.0, 1.0, 1.0])
        p_latent = pyro.sample('p_latent', dist.Dirichlet(concentration))
        pyro.sample("obs", dist.Categorical(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.1, 0.6, 0.3])
    data = dist.Categorical(true_probs).sample(sample_shape=(torch.Size((2000,))))
    nuts_kernel = NUTS(model, jit_compile=True, ignore_jit_warnings=True)
    nuts_kernel.mass_matrix_adapter = ArrowheadMassMatrix()
    mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=100)
    mcmc.run(data)
    samples = mcmc.get_samples()
    posterior = samples["p_latent"]
    assert_equal(posterior.mean(0), true_probs, prec=0.02)
