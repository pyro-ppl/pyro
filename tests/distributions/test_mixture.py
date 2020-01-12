# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro.util import torch_isnan
from tests.common import assert_equal


@pytest.mark.parametrize('sample_shape', [(), (6,), (4, 2)])
@pytest.mark.parametrize('batch_shape', [(), (7,), (5, 3)])
@pytest.mark.parametrize('component1',
                         [dist.Normal(1., 2.), dist.Exponential(2.)],
                         ids=['normal', 'exponential'])
@pytest.mark.parametrize('component0',
                         [dist.Normal(1., 2.), dist.Exponential(2.)],
                         ids=['normal', 'exponential'])
def test_masked_mixture_univariate(component0, component1, sample_shape, batch_shape):
    if batch_shape:
        component0 = component0.expand_by(batch_shape)
        component1 = component1.expand_by(batch_shape)
    mask = torch.empty(batch_shape).bernoulli_(0.5).bool()
    d = dist.MaskedMixture(mask, component0, component1)
    assert d.batch_shape == batch_shape
    assert d.event_shape == ()

    assert d.sample().shape == batch_shape
    assert d.mean.shape == batch_shape
    assert d.variance.shape == batch_shape
    x = d.sample(sample_shape)
    assert x.shape == sample_shape + batch_shape

    log_prob = d.log_prob(x)
    assert log_prob.shape == sample_shape + batch_shape
    assert not torch_isnan(log_prob)
    log_prob_0 = component0.log_prob(x)
    log_prob_1 = component1.log_prob(x)
    mask = mask.expand(sample_shape + batch_shape)
    assert_equal(log_prob[mask], log_prob_1[mask])
    assert_equal(log_prob[~mask], log_prob_0[~mask])


@pytest.mark.parametrize('sample_shape', [(), (6,), (4, 2)])
@pytest.mark.parametrize('batch_shape', [(), (7,), (5, 3)])
def test_masked_mixture_multivariate(sample_shape, batch_shape):
    event_shape = torch.Size((8,))
    component0 = dist.MultivariateNormal(torch.zeros(event_shape), torch.eye(event_shape[0]))
    component1 = dist.Uniform(torch.zeros(event_shape), torch.ones(event_shape)).to_event(1)
    if batch_shape:
        component0 = component0.expand_by(batch_shape)
        component1 = component1.expand_by(batch_shape)
    mask = torch.empty(batch_shape).bernoulli_(0.5).bool()
    d = dist.MaskedMixture(mask, component0, component1)
    assert d.batch_shape == batch_shape
    assert d.event_shape == event_shape

    assert d.sample().shape == batch_shape + event_shape
    assert d.mean.shape == batch_shape + event_shape
    assert d.variance.shape == batch_shape + event_shape
    x = d.sample(sample_shape)
    assert x.shape == sample_shape + batch_shape + event_shape

    log_prob = d.log_prob(x)
    assert log_prob.shape == sample_shape + batch_shape
    assert not torch_isnan(log_prob)
    log_prob_0 = component0.log_prob(x)
    log_prob_1 = component1.log_prob(x)
    mask = mask.expand(sample_shape + batch_shape)
    assert_equal(log_prob[mask], log_prob_1[mask])
    assert_equal(log_prob[~mask], log_prob_0[~mask])


@pytest.mark.parametrize('value_shape', [(), (5, 1, 1, 1), (6, 1, 1, 1, 1)])
@pytest.mark.parametrize('component1_shape', [(), (4, 1, 1), (6, 1, 1, 1, 1)])
@pytest.mark.parametrize('component0_shape', [(), (3, 1), (6, 1, 1, 1, 1)])
@pytest.mark.parametrize('mask_shape', [(), (2,), (6, 1, 1, 1, 1)])
def test_broadcast(mask_shape, component0_shape, component1_shape, value_shape):
    mask = torch.empty(torch.Size(mask_shape)).bernoulli_(0.5).bool()
    component0 = dist.Normal(torch.zeros(component0_shape), 1.)
    component1 = dist.Exponential(torch.ones(component1_shape))
    value = torch.ones(value_shape)

    d = dist.MaskedMixture(mask, component0, component1)
    d_shape = broadcast_shape(mask_shape, component0_shape, component1_shape)
    assert d.batch_shape == d_shape

    log_prob_shape = broadcast_shape(d_shape, value_shape)
    assert d.log_prob(value).shape == log_prob_shape


@pytest.mark.parametrize('event_shape', [(), (2,), (2, 3)])
@pytest.mark.parametrize('batch_shape', [(), (3,), (5, 3)])
@pytest.mark.parametrize('sample_shape', [(), (2,), (4, 2)])
def test_expand(sample_shape, batch_shape, event_shape):
    ones_shape = torch.Size((1,) * len(batch_shape))
    mask = torch.empty(ones_shape).bernoulli_(0.5).bool()
    zero = torch.zeros(ones_shape + event_shape)
    d0 = dist.Uniform(zero - 2, zero + 1).to_event(len(event_shape))
    d1 = dist.Uniform(zero - 1, zero + 2).to_event(len(event_shape))
    d = dist.MaskedMixture(mask, d0, d1)

    assert d.sample().shape == ones_shape + event_shape
    assert d.mean.shape == ones_shape + event_shape
    assert d.variance.shape == ones_shape + event_shape
    assert d.sample(sample_shape).shape == sample_shape + ones_shape + event_shape

    assert d.expand(sample_shape + batch_shape).batch_shape == sample_shape + batch_shape
    assert d.expand(sample_shape + batch_shape).sample().shape == sample_shape + batch_shape + event_shape
    assert d.expand(sample_shape + batch_shape).mean.shape == sample_shape + batch_shape + event_shape
    assert d.expand(sample_shape + batch_shape).variance.shape == sample_shape + batch_shape + event_shape
