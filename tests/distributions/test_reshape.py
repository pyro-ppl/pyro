from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import Variable

from pyro.distributions.torch import Bernoulli


@pytest.mark.parametrize('sample_dim,extra_event_dims',
                         [(s, e) for s in range(4) for e in range(4 + s)])
def test_reshape(sample_dim, extra_event_dims):
    batch_dim = 3
    batch_shape, event_shape = torch.Size((5, 4, 3)), torch.Size()
    sample_shape = torch.Size((8, 7, 6))[3 - sample_dim:]
    shape = sample_shape + batch_shape + event_shape

    dist0 = Bernoulli(Variable(0.5 * torch.ones(batch_shape)))
    dist = dist0.reshape(sample_shape, extra_event_dims)
    sample = dist.sample()
    log_prob = dist.log_prob(sample)
    enum = dist.enumerate_support()

    assert sample.shape == shape
    assert dist.mean.shape == shape
    assert dist.variance.shape == shape
    assert log_prob.shape == shape[:sample_dim + batch_dim - extra_event_dims]
    assert enum.shape == torch.Size((2,)) + shape


@pytest.mark.parametrize('sample_dim,extra_event_dims',
                         [(s, e) for s in range(3) for e in range(3 + s)])
def test_reshape_reshape(sample_dim, extra_event_dims):
    batch_dim = 2
    batch_shape, event_shape = torch.Size((6, 5)), torch.Size((4, 3))
    sample_shape = torch.Size((8, 7))[2 - sample_dim:]
    shape = sample_shape + batch_shape + event_shape

    dist0 = Bernoulli(Variable(0.5 * torch.ones(event_shape)))
    dist1 = dist0.reshape(sample_shape=batch_shape, extra_event_dims=2)
    assert dist1.event_shape == event_shape
    assert dist1.batch_shape == batch_shape

    dist = dist1.reshape(sample_shape, extra_event_dims)
    sample = dist.sample()
    log_prob = dist.log_prob(sample)
    enum = dist.enumerate_support()

    assert sample.shape == shape
    assert dist.mean.shape == shape
    assert dist.variance.shape == shape
    assert log_prob.shape == shape[:sample_dim + batch_dim - extra_event_dims]
    assert enum.shape == torch.Size((2,)) + shape
