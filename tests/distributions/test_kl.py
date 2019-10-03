import pytest
import torch
from torch.distributions import kl_divergence

import pyro.distributions as dist
from pyro.distributions.util import sum_rightmost
from tests.common import assert_close


@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
def test_kl_delta_normal_shape(batch_shape):
    v = torch.randn(batch_shape)
    loc = torch.randn(batch_shape)
    scale = torch.randn(batch_shape).exp()
    p = dist.Delta(v)
    q = dist.Normal(loc, scale)
    assert kl_divergence(p, q).shape == batch_shape


@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize('size', [1, 2, 3])
def test_kl_delta_mvn_shape(batch_shape, size):
    v = torch.randn(batch_shape + (size,))
    p = dist.Delta(v, event_dim=1)

    loc = torch.randn(batch_shape + (size,))
    cov = torch.randn(batch_shape + (size, size))
    cov = cov @ cov.transpose(-1, -2) + 0.01 * torch.eye(size)
    q = dist.MultivariateNormal(loc, covariance_matrix=cov)
    assert kl_divergence(p, q).shape == batch_shape


@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(), (4,), (2, 3)], ids=str)
def test_kl_independent_normal(batch_shape, event_shape):
    shape = batch_shape + event_shape
    p = dist.Normal(torch.randn(shape), torch.randn(shape).exp())
    q = dist.Normal(torch.randn(shape), torch.randn(shape).exp())
    actual = kl_divergence(dist.Independent(p, len(event_shape)),
                           dist.Independent(q, len(event_shape)))
    expected = sum_rightmost(kl_divergence(p, q), len(event_shape))
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize('size', [1, 2, 3])
def test_kl_independent_delta_mvn_shape(batch_shape, size):
    v = torch.randn(batch_shape + (size,))
    p = dist.Independent(dist.Delta(v), 1)

    loc = torch.randn(batch_shape + (size,))
    cov = torch.randn(batch_shape + (size, size))
    cov = cov @ cov.transpose(-1, -2) + 0.01 * torch.eye(size)
    q = dist.MultivariateNormal(loc, covariance_matrix=cov)
    assert kl_divergence(p, q).shape == batch_shape


@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize('size', [1, 2, 3])
def test_kl_independent_normal_mvn(batch_shape, size):
    loc = torch.randn(batch_shape + (size,))
    scale = torch.randn(batch_shape + (size,)).exp()
    p1 = dist.Normal(loc, scale).to_event(1)
    p2 = dist.MultivariateNormal(loc, scale_tril=scale.diag_embed())

    loc = torch.randn(batch_shape + (size,))
    cov = torch.randn(batch_shape + (size, size))
    cov = cov @ cov.transpose(-1, -2) + 0.01 * torch.eye(size)
    q = dist.MultivariateNormal(loc, covariance_matrix=cov)

    actual = kl_divergence(p1, q)
    expected = kl_divergence(p2, q)
    assert_close(actual, expected)
