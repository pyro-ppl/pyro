# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest
import torch
from scipy.integrate.quadpack import IntegrationWarning
from scipy.stats import kstest, levy_stable

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
    d = dist.Stable(alpha, beta)

    def sampler(size):
        # Temporarily increase radius to test hole-patching logic.
        # Scipy doesn't handle values of alpha very close to 1.
        try:
            old = pyro.distributions.stable.RADIUS
            pyro.distributions.stable.RADIUS = 0.02
            x = d.sample([size])
        finally:
            pyro.distributions.stable.RADIUS = old

        # Convert from Nolan's parametrization S^0 to scipy parametrization S.
        if alpha == 1:
            z = x
        else:
            z = x + beta * np.tan(np.pi / 2 * alpha)
        return z

    def cdf(x):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=IntegrationWarning)
            result = levy_stable.cdf(x, alpha, beta)
        # Scipy has only an experimental .cdf() function for alpha=1, beta!=0.
        # It sometimes passes and sometimes xfails.
        if w and alpha == 1 and beta != 0:
            pytest.xfail(reason="scipy.stats.levy_stable.cdf is unstable")
        return result

    stat, pvalue = kstest(sampler, cdf, N=num_samples)
    assert pvalue > 0.1, pvalue


@pytest.mark.parametrize("loc", [0, 1, -1, 2, 2])
@pytest.mark.parametrize("scale", [0.5, 1, 2])
def test_normal(loc, scale):
    num_samples = 100000
    expected = dist.Normal(loc, scale).sample([num_samples])
    actual = dist.Stable(2, 0, scale * 0.5 ** 0.5, loc).sample([num_samples])
    assert_close(actual.mean(), expected.mean(), atol=0.01)
    assert_close(actual.std(), expected.std(), atol=0.01)
