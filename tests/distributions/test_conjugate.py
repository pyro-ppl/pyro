from __future__ import absolute_import, division, print_function

import math

import pytest
import torch

import pyro.distributions as dist
from pyro.distributions import BetaBinomial, GammaPoisson
from tests.common import assert_equal


@pytest.mark.parametrize("dist", [
    BetaBinomial(2., 5., 10.),
    BetaBinomial(torch.tensor([2., 4.]), torch.tensor([5., 8.]), torch.tensor([10., 12.])),
    GammaPoisson(2., 2.),
    GammaPoisson(torch.tensor([6., 2]), torch.tensor([2., 8.])),
])
def test_mean(dist):
    analytic_mean = dist.mean
    num_samples = 500000
    sample_mean = dist.sample((num_samples,)).mean(0)
    assert_equal(sample_mean, analytic_mean, prec=0.01)


@pytest.mark.parametrize("dist", [
    BetaBinomial(2., 5., 10.),
    BetaBinomial(torch.tensor([2., 4.]), torch.tensor([5., 8.]), torch.tensor([10., 12.])),
    GammaPoisson(2., 2.),
    GammaPoisson(torch.tensor([6., 2]), torch.tensor([2., 8.])),
])
def test_variance(dist):
    analytic_var = dist.variance
    num_samples = 500000
    sample_var = dist.sample((num_samples,)).var(0)
    assert_equal(sample_var, analytic_var, prec=0.01)


@pytest.mark.parametrize("dist, values", [
    (BetaBinomial(2., 5., 10), None),
    (BetaBinomial(2., 5., 10), None),
    (GammaPoisson(2., 2.), torch.arange(10.)),
    (GammaPoisson(6., 2.), torch.arange(20.)),
])
def test_log_prob_support(dist, values):
    if values is None:
        values = dist.enumerate_support()
    log_probs = dist.log_prob(values)
    assert_equal(log_probs.logsumexp(0), torch.tensor(0.), prec=0.01)


@pytest.mark.parametrize("total_count", [1, 2, 3, 10])
@pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 3, 1)])
def test_beta_binomial_log_prob(total_count, shape):
    concentration0 = torch.randn(shape).exp()
    concentration1 = torch.randn(shape).exp()
    value = torch.arange(1. + total_count)

    num_samples = 100000
    probs = dist.Beta(concentration1, concentration0).sample((num_samples,))
    log_probs = dist.Binomial(total_count, probs).log_prob(value)
    expected = log_probs.logsumexp(0) - math.log(num_samples)

    actual = BetaBinomial(concentration1, concentration0, total_count).log_prob(value)
    assert_equal(actual, expected, prec=0.05)


@pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 3, 1)])
def test_gamma_poisson_log_prob(shape):
    gamma_conc = torch.randn(shape).exp()
    gamma_rate = torch.randn(shape).exp()
    value = torch.arange(20.)

    num_samples = 200000
    poisson_rate = dist.Gamma(gamma_conc, gamma_rate).sample((num_samples,))
    log_probs = dist.Poisson(poisson_rate).log_prob(value)
    expected = log_probs.logsumexp(0) - math.log(num_samples)
    actual = GammaPoisson(gamma_conc, gamma_rate).log_prob(value)
    err = (expected - actual) / expected
    assert_equal(err, torch.zeros(expected.shape), prec=0.05)
