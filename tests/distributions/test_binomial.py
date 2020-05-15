# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import pyro.distributions as dist
from tests.common import assert_close


@pytest.mark.parametrize("total_count", [10, 100, 1000, 4000])
@pytest.mark.parametrize("prob", [0.01, 0.1, 0.5, 0.9, 0.99])
def test_binomial_approx_sample(total_count, prob):
    sample_shape = (10000,)
    d1 = dist.Binomial(total_count, prob)
    d2 = dist.Binomial(total_count, prob, approx_sample_thresh=200)
    expected = d1.sample(sample_shape)
    actual = d2.sample(sample_shape)

    assert_close(expected.mean(), actual.mean(), rtol=0.05)
    assert_close(expected.std(), actual.std(), rtol=0.05)


@pytest.mark.parametrize("total_count", [10, 100, 1000, 4000])
@pytest.mark.parametrize("concentration1", [0.1, 1.0, 10.])
@pytest.mark.parametrize("concentration0", [0.1, 1.0, 10.])
def test_beta_binomial_approx_sample(concentration1, concentration0, total_count):
    sample_shape = (10000,)
    d1 = dist.BetaBinomial(concentration1, concentration0, total_count)
    d2 = dist.BetaBinomial(concentration1, concentration0, total_count,
                           approx_sample_thresh=200)
    expected = d1.sample(sample_shape)
    actual = d2.sample(sample_shape)

    assert_close(expected.mean(), actual.mean(), rtol=0.1)
    assert_close(expected.std(), actual.std(), rtol=0.1)
