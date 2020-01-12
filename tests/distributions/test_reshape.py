# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.distributions.torch import Bernoulli
from tests.common import assert_equal


def test_sample_shape_order():
    shape12 = torch.Size((1, 2))
    shape34 = torch.Size((3, 4))
    d = Bernoulli(0.5)

    # .expand_by(...) should add dimensions on the left.
    actual = d.expand_by(shape34).expand_by(shape12)
    expected = d.expand_by(shape12 + shape34)
    assert actual.event_shape == expected.event_shape
    assert actual.batch_shape == expected.batch_shape


@pytest.mark.parametrize('batch_dim', [0, 1, 2])
@pytest.mark.parametrize('event_dim', [0, 1, 2])
def test_idempotent(batch_dim, event_dim):
    shape = torch.Size((1, 2, 3, 4))[:batch_dim + event_dim]
    batch_shape = shape[:batch_dim]
    event_shape = shape[batch_dim:]

    # Construct a base dist of desired starting shape.
    dist0 = Bernoulli(0.5).expand_by(shape).to_event(event_dim)
    assert dist0.batch_shape == batch_shape
    assert dist0.event_shape == event_shape

    # Check that an .expand_by() an empty shape is a no-op.
    dist = dist0.expand_by([])
    assert dist.batch_shape == dist0.batch_shape
    assert dist.event_shape == dist0.event_shape


@pytest.mark.parametrize('sample_dim,extra_event_dims',
                         [(s, e) for s in range(4) for e in range(4 + s)])
def test_reshape(sample_dim, extra_event_dims):
    batch_dim = 3
    batch_shape, event_shape = torch.Size((5, 4, 3)), torch.Size()
    sample_shape = torch.Size((8, 7, 6))[3 - sample_dim:]
    shape = sample_shape + batch_shape + event_shape

    # Construct a base dist of desired starting shape.
    dist0 = Bernoulli(0.5 * torch.ones(batch_shape))
    assert dist0.event_shape == event_shape
    assert dist0.batch_shape == batch_shape

    # Check that reshaping has the desired final shape.
    dist = dist0.expand_by(sample_shape).to_event(extra_event_dims)
    sample = dist.sample()
    assert sample.shape == shape
    assert dist.mean.shape == shape
    assert dist.variance.shape == shape
    assert dist.log_prob(sample).shape == shape[:sample_dim + batch_dim - extra_event_dims]

    # Check enumerate support.
    if dist.event_shape:
        with pytest.raises(NotImplementedError):
            dist.enumerate_support()
        with pytest.raises(NotImplementedError):
            dist.enumerate_support(expand=True)
        with pytest.raises(NotImplementedError):
            dist.enumerate_support(expand=False)
    else:
        assert dist.enumerate_support().shape == (2,) + shape
        assert dist.enumerate_support(expand=True).shape == (2,) + shape
        assert dist.enumerate_support(expand=False).shape == (2,) + (1,) * len(sample_shape + batch_shape) + event_shape


@pytest.mark.parametrize('sample_dim,extra_event_dims',
                         [(s, e) for s in range(3) for e in range(3 + s)])
def test_reshape_reshape(sample_dim, extra_event_dims):
    batch_dim = 2
    batch_shape, event_shape = torch.Size((6, 5)), torch.Size((4, 3))
    sample_shape = torch.Size((8, 7))[2 - sample_dim:]
    shape = sample_shape + batch_shape + event_shape

    # Construct a base dist of desired starting shape.
    dist0 = Bernoulli(0.5 * torch.ones(event_shape))
    dist1 = dist0.expand_by(batch_shape).to_event(2)
    assert dist1.event_shape == event_shape
    assert dist1.batch_shape == batch_shape

    # Check that reshaping has the desired final shape.
    dist = dist1.expand_by(sample_shape).to_event(extra_event_dims)
    sample = dist.sample()
    assert sample.shape == shape
    assert dist.mean.shape == shape
    assert dist.variance.shape == shape
    assert dist.log_prob(sample).shape == shape[:sample_dim + batch_dim - extra_event_dims]

    # Check enumerate support.
    if dist.event_shape:
        with pytest.raises(NotImplementedError):
            dist.enumerate_support()
        with pytest.raises(NotImplementedError):
            dist.enumerate_support(expand=True)
        with pytest.raises(NotImplementedError):
            dist.enumerate_support(expand=False)
    else:
        assert dist.enumerate_support().shape == (2,) + shape
        assert dist.enumerate_support(expand=True).shape == (2,) + shape
        assert dist.enumerate_support(expand=False).shape == (2,) + (1,) * len(sample_shape + batch_shape) + event_shape


@pytest.mark.parametrize('sample_dim', [0, 1, 2])
@pytest.mark.parametrize('batch_dim', [0, 1, 2])
@pytest.mark.parametrize('event_dim', [0, 1, 2])
def test_extra_event_dim_overflow(sample_dim, batch_dim, event_dim):
    shape = torch.Size(range(sample_dim + batch_dim + event_dim))
    sample_shape = shape[:sample_dim]
    batch_shape = shape[sample_dim:sample_dim+batch_dim]
    event_shape = shape[sample_dim + batch_dim:]

    # Construct a base dist of desired starting shape.
    dist0 = Bernoulli(0.5).expand_by(batch_shape + event_shape).to_event(event_dim)
    assert dist0.batch_shape == batch_shape
    assert dist0.event_shape == event_shape

    # Check .to_event(...) for valid values.
    for extra_event_dims in range(1 + sample_dim + batch_dim):
        dist = dist0.expand_by(sample_shape).to_event(extra_event_dims)
        assert dist.batch_shape == shape[:sample_dim + batch_dim - extra_event_dims]
        assert dist.event_shape == shape[sample_dim + batch_dim - extra_event_dims:]

    # Check .to_event(...) for invalid values.
    for extra_event_dims in range(1 + sample_dim + batch_dim, 20):
        with pytest.raises(ValueError):
            dist0.expand_by(sample_shape).to_event(extra_event_dims)


def test_independent_entropy():
    dist_univ = Bernoulli(0.5)
    dist_multi = Bernoulli(torch.Tensor([0.5, 0.5])).to_event(1)
    assert_equal(dist_multi.entropy(), 2*dist_univ.entropy())
