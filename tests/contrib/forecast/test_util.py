# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.distributions import transform_to

import pyro.distributions as dist
from pyro.contrib.forecast.util import UNIVARIATE_DISTS, prefix_condition
from tests.ops.gaussian import random_mvn


def random_dist(Dist, shape):
    if Dist is dist.GaussianHMM:
        batch_shape, duration, dim = shape[:-2], shape[-2], shape[-1]
        init_dist = random_dist(dist.Normal, batch_shape + (dim,)).to_event(1)
        trans_mat = torch.randn(batch_shape + (duration, dim, dim))
        trans_dist = random_dist(dist.Normal, batch_shape + (duration, dim)).to_event(1)
        obs_mat = torch.randn(batch_shape + (duration, dim, dim))
        obs_dist = random_dist(dist.Normal, batch_shape + (duration, dim)).to_event(1)
        return Dist(init_dist, trans_mat, trans_dist, obs_mat, obs_dist,
                    duration=duration)
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
@pytest.mark.parametrize("Dist", [
    dist.Bernoulli,
    dist.Beta,
    dist.BetaBinomial,
    dist.Cauchy,
    dist.Dirichlet,
    dist.DirichletMultinomial,
    dist.Exponential,
    dist.Gamma,
    dist.GammaPoisson,
    dist.GaussianHMM,
    dist.InverseGamma,
    dist.Laplace,
    dist.LogNormal,
    dist.MultivariateNormal,
    dist.Normal,
    dist.StudentT,
    dist.ZeroInflatedPoisson,
])
def test_prefix_condition(Dist, batch_shape, t, f, dim):
    duration = t + f
    d = random_dist(Dist, batch_shape + (duration, dim))
    d = d.to_event(2 - d.event_dim)
    data = d.sample()
    expected = d.log_prob(data)
    d2 = prefix_condition(d, data[..., :t, :])
    actual = d2.log_prob(data[..., t:, :])
    actual.shape == expected.shape
