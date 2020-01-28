# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_equal


@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)])
def test_shapes(batch_shape):
    log_factor = torch.randn(batch_shape)

    d = dist.Unit(log_factor=log_factor)
    x = d.sample()
    assert x.shape == batch_shape + (0,)
    assert (d.log_prob(x) == log_factor).all()


@pytest.mark.parametrize('sample_shape', [(), (4,), (3, 2)])
@pytest.mark.parametrize('batch_shape', [(), (7,), (6, 5)])
def test_expand(sample_shape, batch_shape):
    log_factor = torch.randn(batch_shape)
    d1 = dist.Unit(log_factor)
    v1 = d1.sample()

    d2 = d1.expand(sample_shape + batch_shape)
    assert d2.batch_shape == sample_shape + batch_shape
    v2 = d2.sample()
    assert v2.shape == sample_shape + batch_shape + (0,)
    assert_equal(d1.log_prob(v2), d2.log_prob(v1))
