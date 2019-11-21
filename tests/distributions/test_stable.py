import numpy as np
import pytest
import torch
from scipy.stats import levy_stable, kstest

import pyro.distributions as dist


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


@pytest.mark.parametrize("beta", [-0.8, -0.4, 0.0, 0.4, 0.8])
@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8, 0.99, 1.01, 1.2, 1.5, 1.8])
def test_sample(alpha, beta):
    num_samples = 100
    d = dist.Stable(alpha, beta)

    def sampler(size):
        x = d.sample([size])
        # Convert from Nolan's parametrization S^0 to scipy parametrization S.
        z = x + beta * np.tan(np.pi / 2 * alpha)
        return z

    def cdf(x):
        return levy_stable.cdf(x, alpha, beta)

    stat, pvalue = kstest(sampler, cdf, N=num_samples)
    assert pvalue > 0.1, pvalue
