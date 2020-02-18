# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from pyro.contrib.forecast.util import prefix_condition
from tests.ops.gaussian import random_mvn


def random_normal(shape):
    loc = torch.randn(shape)
    scale = torch.rand(shape).exp()
    return dist.Normal(loc, scale)


def random_studentt(shape):
    df = torch.rand(shape).exp()
    loc = torch.randn(shape)
    scale = torch.rand(shape).exp()
    return dist.StudentT(df, loc, scale)


@pytest.mark.parametrize("dim", [1, 7])
@pytest.mark.parametrize("t,f", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(), (6,), (5, 4)])
@pytest.mark.parametrize("Dist", [
    dist.Normal,
    dist.StudentT,
    dist.MultivariateNormal,
    dist.GaussianHMM,
])
def test_prefix_condition(Dist, batch_shape, t, f, dim):
    duration = t + f
    if Dist is dist.GaussianHMM:
        init_dist = random_normal(batch_shape + (dim,)).to_event(1)
        trans_mat = torch.randn(batch_shape + (duration, dim, dim))
        trans_dist = random_normal(batch_shape + (duration, dim)).to_event(1)
        obs_mat = torch.randn(batch_shape + (duration, dim, dim))
        obs_dist = random_normal(batch_shape + (duration, dim)).to_event(1)
        d = Dist(init_dist, trans_mat, trans_dist, obs_mat, obs_dist,
                 duration=duration)
    elif Dist is dist.MultivariateNormal:
        d = random_mvn(batch_shape + (duration,), dim).to_event(1)
    elif Dist is dist.Normal:
        d = random_normal(batch_shape + (duration, dim)).to_event(2)
    elif Dist is dist.StudentT:
        d = random_studentt(batch_shape + (duration, dim)).to_event(2)
    else:
        raise ValueError(Dist.__name__)

    data = torch.randn(batch_shape + (duration, dim))
    expected = d.log_prob(data)
    d2 = prefix_condition(d, data[..., :t, :])
    actual = d2.log_prob(data[..., t:, :])
    actual.shape == expected.shape
