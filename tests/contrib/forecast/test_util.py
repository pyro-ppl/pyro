# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.distributions import transform_to

import pyro.distributions as dist
from pyro.contrib.forecast.util import UNIVARIATE_DISTS, UNIVARIATE_TRANSFORMS, prefix_condition, reshape_batch
from tests.ops.gaussian import random_mvn

DISTS = [
    dist.Bernoulli,
    dist.Beta,
    dist.BetaBinomial,
    dist.Cauchy,
    dist.Dirichlet,
    dist.DirichletMultinomial,
    dist.Exponential,
    dist.FoldedDistribution,
    dist.Gamma,
    dist.GammaPoisson,
    dist.GaussianHMM,
    dist.Geometric,
    dist.IndependentHMM,
    dist.InverseGamma,
    dist.Laplace,
    dist.LinearHMM,
    dist.LogNormal,
    dist.MultivariateNormal,
    dist.NegativeBinomial,
    dist.Normal,
    dist.StudentT,
    dist.ZeroInflatedPoisson,
    dist.ZeroInflatedNegativeBinomial,
]


def random_dist(Dist, shape, transform=None):
    if Dist is dist.FoldedDistribution:
        return Dist(random_dist(dist.Normal, shape))
    elif Dist in (dist.GaussianHMM, dist.LinearHMM):
        batch_shape, duration, obs_dim = shape[:-2], shape[-2], shape[-1]
        hidden_dim = obs_dim + 1
        init_dist = random_dist(dist.Normal, batch_shape + (hidden_dim,)).to_event(1)
        trans_mat = torch.randn(batch_shape + (duration, hidden_dim, hidden_dim))
        trans_dist = random_dist(dist.Normal, batch_shape + (duration, hidden_dim)).to_event(1)
        obs_mat = torch.randn(batch_shape + (duration, hidden_dim, obs_dim))
        obs_dist = random_dist(dist.Normal, batch_shape + (duration, obs_dim)).to_event(1)
        if Dist is dist.LinearHMM and transform is not None:
            obs_dist = dist.TransformedDistribution(obs_dist, transform)
        return Dist(init_dist, trans_mat, trans_dist, obs_mat, obs_dist,
                    duration=duration)
    elif Dist is dist.IndependentHMM:
        batch_shape, duration, obs_dim = shape[:-2], shape[-2], shape[-1]
        base_shape = batch_shape + (obs_dim, duration, 1)
        base_dist = random_dist(dist.GaussianHMM, base_shape)
        return Dist(base_dist)
    elif Dist is dist.MultivariateNormal:
        return random_mvn(shape[:-1], shape[-1])
    else:
        params = {
            name: transform_to(Dist.arg_constraints[name])(torch.rand(shape) - 0.5)
            for name in UNIVARIATE_DISTS[Dist]}
        return Dist(**params)


@pytest.mark.parametrize("dim", [1, 7])
@pytest.mark.parametrize("t,f", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(), (6,), (5, 4)])
@pytest.mark.parametrize("Dist", DISTS)
def test_prefix_condition(Dist, batch_shape, t, f, dim):
    if Dist is dist.LinearHMM:
        pytest.xfail(reason="not implemented")
    duration = t + f
    d = random_dist(Dist, batch_shape + (duration, dim))
    d = d.to_event(2 - d.event_dim)
    data = d.sample()
    expected = d.log_prob(data)
    d2 = prefix_condition(d, data[..., :t, :])
    actual = d2.log_prob(data[..., t:, :])
    actual.shape == expected.shape


@pytest.mark.parametrize("dim", [1, 7])
@pytest.mark.parametrize("duration", [1, 2, 3])
@pytest.mark.parametrize("batch_shape", [(), (6,), (5, 4)])
@pytest.mark.parametrize("Dist", DISTS)
def test_reshape_batch(Dist, batch_shape, duration, dim):
    d = random_dist(Dist, batch_shape + (duration, dim))
    d = d.to_event(2 - d.event_dim)
    assert d.batch_shape == batch_shape
    assert d.event_shape == (duration, dim)

    actual = reshape_batch(d, batch_shape + (1,))
    assert type(actual) is type(d)
    assert actual.batch_shape == batch_shape + (1,)
    assert actual.event_shape == (duration, dim)


@pytest.mark.parametrize("dim", [1, 7])
@pytest.mark.parametrize("duration", [1, 2, 3])
@pytest.mark.parametrize("batch_shape", [(), (6,), (5, 4)])
@pytest.mark.parametrize("transform", list(UNIVARIATE_TRANSFORMS.keys()))
def test_reshape_transform_batch(transform, batch_shape, duration, dim):
    params = {p: torch.rand(batch_shape + (duration, dim))
              for p in UNIVARIATE_TRANSFORMS[transform]}
    t = transform(**params)
    d = random_dist(dist.LinearHMM, batch_shape + (duration, dim), transform=t)
    d = d.to_event(2 - d.event_dim)
    assert d.batch_shape == batch_shape
    assert d.event_shape == (duration, dim)

    actual = reshape_batch(d, batch_shape + (1,))
    assert type(actual) is type(d)
    assert actual.batch_shape == batch_shape + (1,)
    assert actual.event_shape == (duration, dim)

    # test if we have reshape transforms correctly
    assert actual.rsample().shape == actual.shape()
