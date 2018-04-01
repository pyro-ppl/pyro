from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch import tensor

from pyro.distributions.torch import Bernoulli
from tests.common import assert_equal


def checker_mask(shape):
    mask = tensor(0.)
    for size in shape:
        mask = mask.unsqueeze(-1) + torch.arange(size)
    return mask.fmod(2)


@pytest.mark.parametrize('batch_dim,mask_dim',
                         [(b, m) for b in range(3) for m in range(1 + b)])
@pytest.mark.parametrize('event_dim', [0, 1, 2])
def test_mask(batch_dim, event_dim, mask_dim):
    # Construct base distribution.
    shape = torch.Size([2, 3, 4, 5, 6][:batch_dim + event_dim])
    batch_shape = shape[:batch_dim]
    mask_shape = batch_shape[batch_dim - mask_dim:]
    base_dist = Bernoulli(0.1).reshape(shape, event_dim)

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
    assert_equal(dist.log_prob(sample), base_dist.log_prob(sample) * mask)
    assert_equal(dist.score_parts(sample), base_dist.score_parts(sample) * mask, prec=0)
    if not dist.event_shape:
        assert_equal(dist.enumerate_support(), base_dist.enumerate_support())


@pytest.mark.parametrize('batch_shape,mask_shape', [
    ([], [1]),
    ([], [2]),
    ([1], [2]),
    ([2], [3]),
    ([2], [1, 1]),
    ([2, 1], [2]),
])
def test_mask_invalid_shape(batch_shape, mask_shape):
    dist = Bernoulli(0.1).reshape(batch_shape)
    mask = checker_mask(mask_shape)
    with pytest.raises(ValueError):
        dist.mask(mask)
