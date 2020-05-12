# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
from pyro.distributions import CoalescentTimes, CoalescentTimesWithRate
from pyro.distributions.coalescent import CoalescentTimesConstraint, _sample_coalescent_times


@pytest.mark.parametrize("num_leaves", range(2, 30))
def test_sample_is_valid(num_leaves):
    pyro.set_rng_seed(num_leaves)

    # Check with disperse leaves.
    leaf_times = torch.randn(num_leaves)
    coal_times = _sample_coalescent_times(leaf_times)
    assert CoalescentTimesConstraint(leaf_times).check(coal_times)
    assert len(set(coal_times.tolist())) == len(coal_times)

    # Check with simultaneous leaves.
    leaf_times = torch.zeros(num_leaves)
    coal_times = _sample_coalescent_times(leaf_times)
    assert CoalescentTimesConstraint(leaf_times).check(coal_times)
    assert len(set(coal_times.tolist())) == len(coal_times)


@pytest.mark.parametrize("num_steps", [9])
@pytest.mark.parametrize("sample_shape", [(), (7,), (4, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 3, 5, 11])
def test_simple_smoke(num_leaves, num_steps, batch_shape, sample_shape):
    leaf_times = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
    d = CoalescentTimes(leaf_times)
    coal_times = d.sample(sample_shape)
    assert coal_times.shape == sample_shape + batch_shape + (num_leaves-1,)

    actual = d.log_prob(coal_times)
    assert actual.shape == sample_shape + batch_shape


@pytest.mark.parametrize("num_steps", [9])
@pytest.mark.parametrize("sample_shape", [(), (7,), (4, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 3, 5, 11])
def test_with_rate_smoke(num_leaves, num_steps, batch_shape, sample_shape):
    leaf_times = torch.rand(num_leaves).pow(0.5) * num_steps
    rate_grid = torch.rand(batch_shape + (num_steps,))
    d = CoalescentTimesWithRate(leaf_times, rate_grid)
    coal_times = _sample_coalescent_times(
        leaf_times.expand(sample_shape + batch_shape + (-1,)))
    assert coal_times.shape == sample_shape + batch_shape + (num_leaves-1,)

    actual = d.log_prob(coal_times)
    assert actual.shape == sample_shape + batch_shape
