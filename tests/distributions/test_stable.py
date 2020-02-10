# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
import torch
from scipy.integrate.quadpack import IntegrationWarning
from scipy.stats import ks_2samp, kstest, levy_stable

import pyro.distributions as dist
import pyro.distributions.stable
from tests.common import assert_close


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)])
def test_shape(sample_shape, batch_shape):
    stability = torch.empty(batch_shape).uniform_(0, 2).requires_grad_()
    skew = torch.empty(batch_shape).uniform_(-1, 1).requires_grad_()
    scale = torch.randn(batch_shape).exp().requires_grad_()
    loc = torch.randn(batch_shape).requires_grad_()

    d = dist.Stable(stability, skew, scale, loc)
    assert d.batch_shape == batch_shape

    x = d.rsample(sample_shape)
    assert x.shape == sample_shape + batch_shape

    x.sum().backward()


@pytest.mark.parametrize("beta", [-1.0, -0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("alpha", [0.1, 0.4, 0.8, 0.99, 1.0, 1.01, 1.3, 1.7, 2.0])
def test_sample(alpha, beta):
    num_samples = 100
    d = dist.Stable(alpha, beta, coords="S")

    def sampler(size):
        # Temporarily increase radius to test hole-patching logic.
        # Scipy doesn't handle values of alpha very close to 1.
        try:
            old = pyro.distributions.stable.RADIUS
            pyro.distributions.stable.RADIUS = 0.02
            return d.sample([size])
        finally:
            pyro.distributions.stable.RADIUS = old

    def cdf(x):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=IntegrationWarning)
            result = levy_stable.cdf(x, alpha, beta)
        # Scipy has only an experimental .cdf() function for alpha=1, beta!=0.
        # It sometimes passes and sometimes xfails.
        if w and alpha == 1 and beta != 0:
            pytest.xfail(reason="scipy.stats.levy_stable.cdf is unstable")
        return result

    assert kstest(sampler, cdf, N=num_samples).pvalue > 0.1


@pytest.mark.parametrize("beta", [-1.0, -0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("alpha", [
    0.1, 0.4, 0.8, 0.99,
    0.999999, 1.000001,  # scipy sampler is buggy very close to 1
    1.01, 1.3, 1.7, 2.0,
])
def test_sample_2(alpha, beta):
    num_samples = 10000

    d = dist.Stable(alpha, beta, coords="S")
    # Temporarily increase radius to test hole-patching logic.
    # Scipy doesn't handle values of alpha very close to 1.
    try:
        old = pyro.distributions.stable.RADIUS
        pyro.distributions.stable.RADIUS = 0.02
        actual = d.sample([num_samples])
    finally:
        pyro.distributions.stable.RADIUS = old
    actual = d.sample([num_samples])

    expected = levy_stable.rvs(alpha, beta, size=num_samples)

    assert ks_2samp(expected, actual).pvalue > 0.05


@pytest.mark.parametrize("loc", [0, 1, -1, 2, 2])
@pytest.mark.parametrize("scale", [0.5, 1, 2])
def test_normal(loc, scale):
    num_samples = 100000
    expected = dist.Normal(loc, scale).sample([num_samples])
    actual = dist.Stable(2, 0, scale * 0.5 ** 0.5, loc).sample([num_samples])
    assert_close(actual.mean(), expected.mean(), atol=0.01)
    assert_close(actual.std(), expected.std(), atol=0.01)


@pytest.mark.parametrize("skew0", [-0.9, -0.5, 0.0, 0.5, 0.9])
@pytest.mark.parametrize("skew1", [-0.9, -0.5, 0.0, 0.5, 0.9])
@pytest.mark.parametrize("scale0,scale1", [(0.1, 0.9), (0.2, 0.8), (0.4, 0.6), (0.5, 0.5)])
@pytest.mark.parametrize("stability", [0.5, 0.99, 1.01, 1.5, 1.9])
def test_additive(stability, skew0, skew1, scale0, scale1):
    num_samples = 10000
    d0 = dist.Stable(stability, skew0, scale0, coords="S")
    d1 = dist.Stable(stability, skew1, scale1, coords="S")
    expected = d0.sample([num_samples]) + d1.sample([num_samples])

    scale = (scale0 ** stability + scale1 ** stability) ** (1 / stability)
    skew = ((skew0 * scale0 ** stability + skew1 * scale1 ** stability) /
            (scale0 ** stability + scale1 ** stability))
    d = dist.Stable(stability, skew, scale, coords="S")
    actual = d.sample([num_samples])

    assert ks_2samp(expected, actual).pvalue > 0.05


@pytest.mark.parametrize("scale", [0.5, 1.5])
@pytest.mark.parametrize("skew", [-0.5, 0.0, 0.5, 0.9])
@pytest.mark.parametrize("stability", [0.5, 1.0, 1.7, 2.0])
@pytest.mark.parametrize("coords", ["S0", "S"])
def test_mean(stability, skew, scale, coords):
    loc = torch.randn(10)
    d = dist.Stable(stability, skew, scale, loc, coords=coords)
    if stability <= 1:
        assert torch.isnan(d.mean).all()
    else:
        expected = d.sample((100000,)).mean(0)
        assert_close(d.mean, expected, atol=0.1)


@pytest.mark.parametrize("scale", [0.5, 1.5])
@pytest.mark.parametrize("stability", [1.7, 2.0])
def test_variance(stability, scale):
    skew = dist.Uniform(-1, 1).sample((10,))
    loc = torch.randn(10)
    d = dist.Stable(stability, skew, scale, loc)
    if stability < 2:
        assert torch.isinf(d.variance).all()
    else:
        expected = d.sample((100000,)).var(0)
        assert_close(d.variance, expected, rtol=0.02)
