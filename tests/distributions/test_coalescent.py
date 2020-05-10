# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.distributions.coalescent import CoalescentTimesWithRate


@pytest.mark.parametrize("num_steps", [9])
@pytest.mark.parametrize("sample_shape", [(), (7,), (4, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 3, 5, 11])
def test_shape(num_leaves, num_steps, batch_shape, sample_shape):
    leaf_times = torch.rand(num_leaves) * num_steps
    rate_grid = torch.rand(batch_shape + (num_steps,))
    d = CoalescentTimesWithRate(leaf_times, rate_grid)

    shape = sample_shape + (1,) * len(batch_shape) + (num_leaves - 1,)
    coal_times = torch.rand(shape) * num_steps
    actual = d.log_prob(coal_times)
    assert actual.shape == sample_shape + batch_shape
