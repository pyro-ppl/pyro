# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.distributions import kl_divergence, transforms

import pyro.distributions as dist
from pyro.distributions.util import sum_rightmost
from tests.common import assert_close


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
def test_kl_delta_normal_shape(batch_shape):
    v = torch.randn(batch_shape)
    loc = torch.randn(batch_shape)
    scale = torch.randn(batch_shape).exp()
    p = dist.Delta(v)
    q = dist.Normal(loc, scale)
    assert kl_divergence(p, q).shape == batch_shape


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("size", [1, 2, 3])
def test_kl_delta_mvn_shape(batch_shape, size):
    v = torch.randn(batch_shape + (size,))
    p = dist.Delta(v, event_dim=1)

    loc = torch.randn(batch_shape + (size,))
    cov = torch.randn(batch_shape + (size, size))
    cov = cov @ cov.transpose(-1, -2) + 0.01 * torch.eye(size)
    q = dist.MultivariateNormal(loc, covariance_matrix=cov)
    assert kl_divergence(p, q).shape == batch_shape


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("event_shape", [(), (4,), (2, 3)], ids=str)
def test_kl_independent_normal(batch_shape, event_shape):
    shape = batch_shape + event_shape
    p = dist.Normal(torch.randn(shape), torch.randn(shape).exp())
    q = dist.Normal(torch.randn(shape), torch.randn(shape).exp())
    actual = kl_divergence(
        dist.Independent(p, len(event_shape)), dist.Independent(q, len(event_shape))
    )
    expected = sum_rightmost(kl_divergence(p, q), len(event_shape))
    assert_close(actual, expected)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("size", [1, 2, 3])
def test_kl_independent_delta_mvn_shape(batch_shape, size):
    v = torch.randn(batch_shape + (size,))
    p = dist.Independent(dist.Delta(v), 1)

    loc = torch.randn(batch_shape + (size,))
    cov = torch.randn(batch_shape + (size, size))
    cov = cov @ cov.transpose(-1, -2) + 0.01 * torch.eye(size)
    q = dist.MultivariateNormal(loc, covariance_matrix=cov)
    assert kl_divergence(p, q).shape == batch_shape


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("size", [1, 2, 3])
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


@pytest.mark.parametrize("shape", [(5,), (4, 5), (2, 3, 5)], ids=str)
@pytest.mark.parametrize("event_dim", [0, 1])
@pytest.mark.parametrize(
    "transform", [transforms.ExpTransform(), transforms.StickBreakingTransform()]
)
def test_kl_transformed_transformed(shape, event_dim, transform):
    p_base = dist.Normal(torch.zeros(shape), torch.ones(shape)).to_event(event_dim)
    q_base = dist.Normal(torch.ones(shape) * 2, torch.ones(shape)).to_event(event_dim)
    p = dist.TransformedDistribution(p_base, transform)
    q = dist.TransformedDistribution(q_base, transform)
    kl = kl_divergence(q, p)
    expected_shape = shape[:-1] if max(transform.event_dim, event_dim) == 1 else shape
    assert kl.shape == expected_shape


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 1)])
@pytest.mark.parametrize("independent", [False, True])
def test_kl_delta_preserves_log_density(batch_shape, independent):
    value = torch.ones(batch_shape + (3,))
    density_shape = value.shape if independent else batch_shape
    log_density = torch.full(density_shape, 0.4, requires_grad=True)
    if independent:
        p = dist.Delta(value, log_density).to_event(1)
    else:
        p = dist.Delta(value, log_density, event_dim=1)
    q = dist.MultivariateNormal(torch.zeros_like(value), torch.eye(3))

    actual = kl_divergence(p, q)
    expected = p.log_prob(value) - q.log_prob(value)
    assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.sum(), log_density)[0]
    assert_close(gradient, torch.ones_like(log_density))


@pytest.mark.parametrize("prior", ["lognormal", "mvn"])
def test_kl_delta_jacobian_in_mean_field_elbo(prior):
    import pyro
    from pyro.infer import Trace_ELBO, TraceMeanField_ELBO

    loc = torch.tensor([0.4, -0.2], requires_grad=True)

    def model():
        if prior == "lognormal":
            fn = dist.LogNormal(torch.zeros(2), torch.ones(2)).to_event(1)
        else:
            fn = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
        pyro.sample("x", fn)

    def guide():
        z = pyro.sample(
            "aux", dist.Normal(loc, 0.7).to_event(1), infer={"is_auxiliary": True}
        )
        if prior == "lognormal":
            fn = dist.Delta(z.exp(), log_density=-z.sum(), event_dim=1)
        else:
            fn = dist.Delta(z.exp(), log_density=-z).to_event(1)
        pyro.sample("x", fn)

    results = []
    for estimator in (Trace_ELBO, TraceMeanField_ELBO):
        pyro.set_rng_seed(2026)
        loss = estimator().differentiable_loss(model, guide)
        gradient = torch.autograd.grad(loss, loc)[0]
        results.append((loss, gradient))
    assert_close(results[0][0], results[1][0])
    assert_close(results[0][1], results[1][1])
