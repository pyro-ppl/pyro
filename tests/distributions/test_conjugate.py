# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import pyro.distributions as dist
from pyro.distributions import BetaBinomial, DirichletMultinomial, GammaPoisson
from tests.common import assert_close


@pytest.mark.parametrize("dist", [
    BetaBinomial(2., 5., 10.),
    BetaBinomial(torch.tensor([2., 4.]), torch.tensor([5., 8.]), torch.tensor([10., 12.])),
    DirichletMultinomial(torch.tensor([0.5, 1.0, 2.0]), 5),
    DirichletMultinomial(torch.tensor([[0.5, 1.0, 2.0], [0.2, 0.5, 0.8]]), torch.tensor(10.)),
    GammaPoisson(2., 2.),
    GammaPoisson(torch.tensor([6., 2]), torch.tensor([2., 8.])),
])
def test_mean(dist):
    analytic_mean = dist.mean
    num_samples = 500000
    sample_mean = dist.sample((num_samples,)).mean(0)
    assert_close(sample_mean, analytic_mean, atol=0.01)


@pytest.mark.parametrize("dist", [
    BetaBinomial(2., 5., 10.),
    BetaBinomial(torch.tensor([2., 4.]), torch.tensor([5., 8.]), torch.tensor([10., 12.])),
    DirichletMultinomial(torch.tensor([0.5, 1.0, 2.0]), 5),
    DirichletMultinomial(torch.tensor([[0.5, 1.0, 2.0], [0.2, 0.5, 0.8]]), torch.tensor(10.)),
    GammaPoisson(2., 2.),
    GammaPoisson(torch.tensor([6., 2]), torch.tensor([2., 8.])),
])
def test_variance(dist):
    analytic_var = dist.variance
    num_samples = 500000
    sample_var = dist.sample((num_samples,)).var(0)
    assert_close(sample_var, analytic_var, rtol=0.01)


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
    assert_close(log_probs.logsumexp(0), torch.tensor(0.), atol=0.01)


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
    assert_close(actual, expected, rtol=0.02)


@pytest.mark.parametrize("total_count", [1, 2, 3, 10])
@pytest.mark.parametrize("batch_shape", [(1,), (3, 1), (2, 3, 1)])
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_dirichlet_multinomial_log_prob(total_count, batch_shape, is_sparse):
    event_shape = (3,)
    concentration = torch.rand(batch_shape + event_shape).exp()
    # test on one-hots
    value = total_count * torch.eye(3).reshape(event_shape + (1,) * len(batch_shape) + event_shape)

    num_samples = 100000
    probs = dist.Dirichlet(concentration).sample((num_samples, 1))
    log_probs = dist.Multinomial(total_count, probs).log_prob(value)
    assert log_probs.shape == (num_samples,) + event_shape + batch_shape
    expected = log_probs.logsumexp(0) - math.log(num_samples)

    actual = DirichletMultinomial(concentration, total_count, is_sparse).log_prob(value)
    assert_close(actual, expected, atol=0.05)


@pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 3, 1)])
def test_gamma_poisson_log_prob(shape):
    gamma_conc = torch.randn(shape).exp()
    gamma_rate = torch.randn(shape).exp()
    value = torch.arange(20.)

    num_samples = 300000
    poisson_rate = dist.Gamma(gamma_conc, gamma_rate).sample((num_samples,))
    log_probs = dist.Poisson(poisson_rate).log_prob(value)
    expected = log_probs.logsumexp(0) - math.log(num_samples)
    actual = GammaPoisson(gamma_conc, gamma_rate).log_prob(value)
    assert_close(actual, expected, rtol=0.05)
