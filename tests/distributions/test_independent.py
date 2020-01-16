# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.distributions.utils import _sum_rightmost

import pyro.distributions as dist
from pyro.util import torch_isnan
from tests.common import assert_equal


@pytest.mark.parametrize('sample_shape', [(), (6,), (4, 2)])
@pytest.mark.parametrize('batch_shape', [(), (7,), (5, 3), (5, 3, 2)])
@pytest.mark.parametrize('reinterpreted_batch_ndims', [0, 1, 2, 3])
@pytest.mark.parametrize('base_dist',
                         [dist.Normal(1., 2.), dist.Exponential(2.),
                          dist.MultivariateNormal(torch.zeros(2), torch.eye(2))],
                         ids=['normal', 'exponential', 'mvn'])
def test_independent(base_dist, sample_shape, batch_shape, reinterpreted_batch_ndims):
    if batch_shape:
        base_dist = base_dist.expand_by(batch_shape)
    if reinterpreted_batch_ndims > len(base_dist.batch_shape):
        with pytest.raises(ValueError):
            d = dist.Independent(base_dist, reinterpreted_batch_ndims)
    else:
        d = dist.Independent(base_dist, reinterpreted_batch_ndims)
        assert d.batch_shape == batch_shape[:len(batch_shape) - reinterpreted_batch_ndims]
        assert d.event_shape == batch_shape[len(batch_shape) - reinterpreted_batch_ndims:] + base_dist.event_shape

        assert d.sample().shape == batch_shape + base_dist.event_shape
        assert d.mean.shape == batch_shape + base_dist.event_shape
        assert d.variance.shape == batch_shape + base_dist.event_shape
        x = d.sample(sample_shape)
        assert x.shape == sample_shape + d.batch_shape + d.event_shape

        log_prob = d.log_prob(x)
        assert log_prob.shape == sample_shape + batch_shape[:len(batch_shape) - reinterpreted_batch_ndims]
        assert not torch_isnan(log_prob)
        log_prob_0 = base_dist.log_prob(x)
        assert_equal(log_prob, _sum_rightmost(log_prob_0, reinterpreted_batch_ndims))


@pytest.mark.parametrize('base_dist',
                         [dist.Normal(1., 2.), dist.Exponential(2.),
                          dist.MultivariateNormal(torch.zeros(2), torch.eye(2))],
                         ids=['normal', 'exponential', 'mvn'])
def test_to_event(base_dist):
    base_dist = base_dist.expand([2, 3])
    d = base_dist
    expected_event_dim = d.event_dim

    d = d.to_event(0)
    assert d is base_dist

    d = d.to_event(1)
    expected_event_dim += 1
    assert d.event_dim == expected_event_dim
    assert d.base_dist is base_dist

    d = d.to_event(0)
    assert d.event_dim == expected_event_dim
    assert d.base_dist is base_dist

    d = d.to_event(1)
    expected_event_dim += 1
    assert d.event_dim == expected_event_dim
    assert d.base_dist is base_dist

    d = d.to_event(0)
    assert d.event_dim == expected_event_dim
    assert d.base_dist is base_dist

    d = d.to_event(-1)
    expected_event_dim += -1
    assert d.event_dim == expected_event_dim
    assert d.base_dist is base_dist

    d = d.to_event(0)
    assert d.event_dim == expected_event_dim
    assert d.base_dist is base_dist

    d = d.to_event(-1)
    expected_event_dim += -1
    assert d is base_dist


@pytest.mark.parametrize('event_shape', [(), (2,), (2, 3)])
@pytest.mark.parametrize('batch_shape', [(), (3,), (5, 3)])
@pytest.mark.parametrize('sample_shape', [(), (2,), (4, 2)])
def test_expand(sample_shape, batch_shape, event_shape):
    ones_shape = torch.Size((1,) * len(batch_shape))
    zero = torch.zeros(ones_shape + event_shape)
    d0 = dist.Uniform(zero - 2, zero + 1).to_event(len(event_shape))

    assert d0.sample().shape == ones_shape + event_shape
    assert d0.mean.shape == ones_shape + event_shape
    assert d0.variance.shape == ones_shape + event_shape
    assert d0.sample(sample_shape).shape == sample_shape + ones_shape + event_shape

    assert d0.expand(sample_shape + batch_shape).batch_shape == sample_shape + batch_shape
    assert d0.expand(sample_shape + batch_shape).sample().shape == sample_shape + batch_shape + event_shape
    assert d0.expand(sample_shape + batch_shape).mean.shape == sample_shape + batch_shape + event_shape
    assert d0.expand(sample_shape + batch_shape).variance.shape == sample_shape + batch_shape + event_shape

    base_dist = dist.MultivariateNormal(torch.zeros(2).expand(*(event_shape + (2,))),
                                        torch.eye(2).expand(*(event_shape + (2, 2))))
    if len(event_shape) > len(base_dist.batch_shape):
        with pytest.raises(ValueError):
            base_dist.to_event(len(event_shape)).expand(batch_shape)
    else:
        expanded = base_dist.to_event(len(event_shape)).expand(batch_shape)
        expanded_batch_ndims = getattr(expanded, 'reinterpreted_batch_ndims', 0)
        assert expanded.batch_shape == batch_shape
        assert expanded.event_shape == (base_dist.batch_shape[len(base_dist.batch_shape) -
                                                              expanded_batch_ndims:] +
                                        base_dist.event_shape)
