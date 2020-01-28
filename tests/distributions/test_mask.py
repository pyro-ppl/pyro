# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import tensor
from torch.distributions import kl_divergence

from pyro.distributions.util import broadcast_shape
from pyro.distributions.torch import Bernoulli, Normal
from pyro.distributions.util import scale_and_mask
from tests.common import assert_equal


def checker_mask(shape):
    mask = tensor(0.)
    for size in shape:
        mask = mask.unsqueeze(-1) + torch.arange(float(size))
    return mask.fmod(2).bool()


@pytest.mark.parametrize('batch_dim,mask_dim',
                         [(b, m) for b in range(3) for m in range(1 + b)])
@pytest.mark.parametrize('event_dim', [0, 1, 2])
def test_mask(batch_dim, event_dim, mask_dim):
    # Construct base distribution.
    shape = torch.Size([2, 3, 4, 5, 6][:batch_dim + event_dim])
    batch_shape = shape[:batch_dim]
    mask_shape = batch_shape[batch_dim - mask_dim:]
    base_dist = Bernoulli(0.1).expand_by(shape).to_event(event_dim)

    # Construct masked distribution.
    mask = checker_mask(mask_shape)
    dist = base_dist.mask(mask)

    # Check shape.
    sample = base_dist.sample()
    assert dist.batch_shape == base_dist.batch_shape
    assert dist.event_shape == base_dist.event_shape
    assert sample.shape == sample.shape
    assert dist.log_prob(sample).shape == base_dist.log_prob(sample).shape

    # Check values.
    assert_equal(dist.mean, base_dist.mean)
    assert_equal(dist.variance, base_dist.variance)
    assert_equal(dist.log_prob(sample),
                 scale_and_mask(base_dist.log_prob(sample), mask=mask))
    assert_equal(dist.score_parts(sample),
                 base_dist.score_parts(sample).scale_and_mask(mask=mask), prec=0)
    if not dist.event_shape:
        assert_equal(dist.enumerate_support(), base_dist.enumerate_support())
        assert_equal(dist.enumerate_support(expand=True), base_dist.enumerate_support(expand=True))
        assert_equal(dist.enumerate_support(expand=False), base_dist.enumerate_support(expand=False))


@pytest.mark.parametrize("mask", [False, True, torch.tensor(False), torch.tensor(True)])
def test_mask_type(mask):
    p = Normal(torch.randn(2, 2), torch.randn(2, 2).exp())
    p_masked = p.mask(mask)
    if isinstance(mask, bool):
        mask = torch.tensor(mask)

    x = p.sample()
    actual = p_masked.log_prob(x)
    expected = p.log_prob(x) * mask.float()
    assert_equal(actual, expected)

    actual = p_masked.score_parts(x)
    expected = p.score_parts(x)
    for a, e in zip(actual, expected):
        if isinstance(e, torch.Tensor):
            e = e * mask.float()
        assert_equal(a, e)


@pytest.mark.parametrize("event_shape", [(), (4,)])
@pytest.mark.parametrize("dist_shape", [(), (3,), (2, 1), (2, 3)])
@pytest.mark.parametrize("mask_shape", [(), (3,), (2, 1), (2, 3)])
def test_broadcast(event_shape, dist_shape, mask_shape):
    mask = torch.empty(torch.Size(mask_shape)).bernoulli_(0.5).bool()
    base_dist = Normal(torch.zeros(dist_shape + event_shape), 1.)
    base_dist = base_dist.to_event(len(event_shape))
    assert base_dist.batch_shape == dist_shape
    assert base_dist.event_shape == event_shape

    d = base_dist.mask(mask)
    d_shape = broadcast_shape(mask.shape, base_dist.batch_shape)
    assert d.batch_shape == d_shape
    assert d.event_shape == event_shape


def test_kl_divergence():
    mask = torch.tensor([[0, 1], [1, 1]]).bool()
    p = Normal(torch.randn(2, 2), torch.randn(2, 2).exp())
    q = Normal(torch.randn(2, 2), torch.randn(2, 2).exp())
    expected = kl_divergence(p.to_event(2), q.to_event(2))
    actual = (kl_divergence(p.mask(mask).to_event(2),
                            q.mask(mask).to_event(2)) +
              kl_divergence(p.mask(~mask).to_event(2),
                            q.mask(~mask).to_event(2)))
    assert_equal(actual, expected)


@pytest.mark.parametrize("p_mask", [False, True, torch.tensor(False), torch.tensor(True)])
@pytest.mark.parametrize("q_mask", [False, True, torch.tensor(False), torch.tensor(True)])
def test_kl_divergence_type(p_mask, q_mask):
    p = Normal(torch.randn(2, 2), torch.randn(2, 2).exp())
    q = Normal(torch.randn(2, 2), torch.randn(2, 2).exp())
    mask = ((torch.tensor(p_mask) if isinstance(p_mask, bool) else p_mask) &
            (torch.tensor(q_mask) if isinstance(q_mask, bool) else q_mask)).expand(2, 2)

    expected = kl_divergence(p, q)
    expected[~mask] = 0

    actual = kl_divergence(p.mask(p_mask), q.mask(q_mask))
    if p_mask is False or q_mask is False:
        assert isinstance(actual, float) and actual == 0.
    else:
        assert_equal(actual, expected)


class NormalBomb(Normal):
    def log_prob(self, value):
        raise ValueError("Should not be called")

    def score_parts(self, value):
        raise ValueError("Should not be called")


@pytest.mark.parametrize("shape", [None, (), (4,), (3, 2)], ids=str)
def test_mask_noop(shape):
    d = NormalBomb(0, 1).mask(False)
    if shape is not None:
        d = d.expand(shape)
    x = d.sample()

    actual = d.log_prob(x)
    assert_equal(actual, torch.zeros(shape if shape else ()))

    actual = d.score_parts(x)
    assert_equal(actual.log_prob, torch.zeros(shape if shape else ()))
