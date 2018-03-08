from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.distributions.torch import Bernoulli


def test_sample_shape_order():
    shape12 = torch.Size((1, 2))
    shape34 = torch.Size((3, 4))
    d = Bernoulli(0.5)

    # .reshape(sample_shape=...) should add dimensions on the left.
    actual = d.reshape(shape34).reshape(shape12)
    expected = d.reshape(shape12 + shape34)
    assert actual.event_shape == expected.event_shape
    assert actual.batch_shape == expected.batch_shape


@pytest.mark.parametrize('batch_dim', [0, 1, 2])
@pytest.mark.parametrize('event_dim', [0, 1, 2])
def test_idempotent(batch_dim, event_dim):
    shape = torch.Size((1, 2, 3, 4))[:batch_dim + event_dim]
    batch_shape = shape[:batch_dim]
    event_shape = shape[batch_dim:]

    # Construct a base dist of desired starting shape.
    dist0 = Bernoulli(0.5).reshape(sample_shape=shape, extra_event_dims=event_dim)
    assert dist0.batch_shape == batch_shape
    assert dist0.event_shape == event_shape

    # Check that an empty .reshape() is a no-op.
    dist = dist0.reshape()
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
    dist = dist0.reshape(sample_shape, extra_event_dims)
    sample = dist.sample()
    assert sample.shape == shape
    assert dist.mean.shape == shape
    assert dist.variance.shape == shape
    assert dist.log_prob(sample).shape == shape[:sample_dim + batch_dim - extra_event_dims]

    # Check enumerate support.
    if dist.event_shape:
        with pytest.raises(NotImplementedError):
            dist.enumerate_support()
    else:
        assert dist.enumerate_support().shape == torch.Size((2,)) + shape


@pytest.mark.parametrize('sample_dim,extra_event_dims',
                         [(s, e) for s in range(3) for e in range(3 + s)])
def test_reshape_reshape(sample_dim, extra_event_dims):
    batch_dim = 2
    batch_shape, event_shape = torch.Size((6, 5)), torch.Size((4, 3))
    sample_shape = torch.Size((8, 7))[2 - sample_dim:]
    shape = sample_shape + batch_shape + event_shape

    # Construct a base dist of desired starting shape.
    dist0 = Bernoulli(0.5 * torch.ones(event_shape))
    dist1 = dist0.reshape(sample_shape=batch_shape, extra_event_dims=2)
    assert dist1.event_shape == event_shape
    assert dist1.batch_shape == batch_shape

    # Check that reshaping has the desired final shape.
    dist = dist1.reshape(sample_shape, extra_event_dims)
    sample = dist.sample()
    assert sample.shape == shape
    assert dist.mean.shape == shape
    assert dist.variance.shape == shape
    assert dist.log_prob(sample).shape == shape[:sample_dim + batch_dim - extra_event_dims]

    # Check enumerate support.
    if dist.event_shape:
        with pytest.raises(NotImplementedError):
            dist.enumerate_support()
    else:
        assert dist.enumerate_support().shape == torch.Size((2,)) + shape


@pytest.mark.parametrize('sample_dim', [0, 1, 2])
@pytest.mark.parametrize('batch_dim', [0, 1, 2])
@pytest.mark.parametrize('event_dim', [0, 1, 2])
def test_extra_event_dim_overflow(sample_dim, batch_dim, event_dim):
    shape = torch.Size(range(sample_dim + batch_dim + event_dim))
    sample_shape = shape[:sample_dim]
    batch_shape = shape[sample_dim:sample_dim+batch_dim]
    event_shape = shape[sample_dim + batch_dim:]

    # Construct a base dist of desired starting shape.
    dist0 = Bernoulli(0.5).reshape(sample_shape=batch_shape + event_shape, extra_event_dims=event_dim)
    assert dist0.batch_shape == batch_shape
    assert dist0.event_shape == event_shape

    # Check .reshape(extra_event_dims=...) for valid values.
    for extra_event_dims in range(1 + sample_dim + batch_dim):
        dist = dist0.reshape(sample_shape=sample_shape, extra_event_dims=extra_event_dims)
        assert dist.batch_shape == shape[:sample_dim + batch_dim - extra_event_dims]
        assert dist.event_shape == shape[sample_dim + batch_dim - extra_event_dims:]

    # Check .reshape(extra_event_dims=...) for invalid values.
    for extra_event_dims in range(1 + sample_dim + batch_dim, 20):
        with pytest.raises(ValueError):
            dist0.reshape(sample_shape=sample_shape, extra_event_dims=extra_event_dims)
