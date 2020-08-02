# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.distributions import constraints, transform_to

import pyro.distributions as dist
from tests.common import assert_equal


@pytest.mark.parametrize("constraint", [
    constraints.real,
    constraints.positive,
    constraints.unit_interval,
], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("event_shape", [(), (4,), (3, 2)], ids=str)
def test_improper_uniform(constraint, batch_shape, event_shape):
    d = dist.ImproperUniform(constraint, batch_shape, event_shape)

    value = transform_to(constraint)(torch.randn(batch_shape + event_shape))
    assert_equal(d.log_prob(value), torch.zeros(batch_shape))

    with pytest.raises(NotImplementedError):
        d.sample()
    with pytest.raises(NotImplementedError):
        d.sample(sample_shape=(5, 6))
