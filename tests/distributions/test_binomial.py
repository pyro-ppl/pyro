# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from pyro.contrib.epidemiology.distributions import set_approx_log_prob_tol, set_approx_sample_thresh
from tests.common import assert_close


@pytest.mark.parametrize("total_count", [10, 100, 1000, 4000])
@pytest.mark.parametrize("prob", [0.01, 0.1, 0.5, 0.9, 0.99])
def test_binomial_approx_sample(total_count, prob):
    sample_shape = (10000,)
    d = dist.Binomial(total_count, prob)
    expected = d.sample(sample_shape)
    with set_approx_sample_thresh(200):
        actual = d.sample(sample_shape)

    assert_close(expected.mean(), actual.mean(), rtol=0.05)
    assert_close(expected.std(), actual.std(), rtol=0.05)


@pytest.mark.parametrize("total_count", [10, 100, 1000, 4000])
@pytest.mark.parametrize("concentration1", [0.1, 1.0, 10.])
@pytest.mark.parametrize("concentration0", [0.1, 1.0, 10.])
def test_beta_binomial_approx_sample(concentration1, concentration0, total_count):
    sample_shape = (10000,)
    d = dist.BetaBinomial(concentration1, concentration0, total_count)
    expected = d.sample(sample_shape)
    with set_approx_sample_thresh(200):
        actual = d.sample(sample_shape)

    assert_close(expected.mean(), actual.mean(), rtol=0.1)
    assert_close(expected.std(), actual.std(), rtol=0.1)


@pytest.mark.parametrize("tol", [
    1e-8, 1e-6, 1e-4, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.1, 1.,
])
def test_binomial_approx_log_prob(tol):
    logits = torch.linspace(-10., 10., 100)
    k = torch.arange(100.).unsqueeze(-1)
    n_minus_k = torch.arange(100.).unsqueeze(-1).unsqueeze(-1)
    n = k + n_minus_k

    expected = torch.distributions.Binomial(n, logits=logits).log_prob(k)
    with set_approx_log_prob_tol(tol):
        actual = dist.Binomial(n, logits=logits).log_prob(k)

    assert_close(actual, expected, atol=tol)
