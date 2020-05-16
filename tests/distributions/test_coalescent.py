# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
from pyro.distributions import CoalescentTimes, CoalescentTimesWithRate
from pyro.distributions.coalescent import CoalescentRateLikelihood, CoalescentTimesConstraint, _sample_coalescent_times
from pyro.distributions.util import broadcast_shape
from tests.common import assert_close


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
@pytest.mark.parametrize("sample_shape", [(), (6,), (4, 5)], ids=str)
@pytest.mark.parametrize("rate_grid_shape", [(), (2,), (3, 1), (3, 2)], ids=str)
@pytest.mark.parametrize("leaf_times_shape", [(), (2,), (3, 1), (3, 2)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 7, 11])
def test_with_rate_smoke(num_leaves, num_steps, leaf_times_shape, rate_grid_shape, sample_shape):
    batch_shape = broadcast_shape(leaf_times_shape, rate_grid_shape)
    leaf_times = torch.rand(leaf_times_shape + (num_leaves,)).pow(0.5) * num_steps
    rate_grid = torch.rand(rate_grid_shape + (num_steps,))
    d = CoalescentTimesWithRate(leaf_times, rate_grid)
    coal_times = _sample_coalescent_times(
        leaf_times.expand(sample_shape + batch_shape + (-1,)))
    assert coal_times.shape == sample_shape + batch_shape + (num_leaves-1,)

    actual = d.log_prob(coal_times)
    assert actual.shape == sample_shape + batch_shape


@pytest.mark.parametrize("num_steps", [9])
@pytest.mark.parametrize("sample_shape", [(), (5,)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 7, 11])
def test_log_prob_unit_rate(num_leaves, num_steps, batch_shape, sample_shape):
    leaf_times = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
    d1 = CoalescentTimes(leaf_times)

    rate_grid = torch.ones(batch_shape + (num_steps,))
    d2 = CoalescentTimesWithRate(leaf_times, rate_grid)

    coal_times = d1.sample(sample_shape)
    assert_close(d1.log_prob(coal_times), d2.log_prob(coal_times))


@pytest.mark.parametrize("num_steps", [9])
@pytest.mark.parametrize("sample_shape", [(), (5,)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 7, 11])
def test_log_prob_constant_rate(num_leaves, num_steps, batch_shape, sample_shape):
    rate = torch.randn(batch_shape).exp()
    rate_grid = rate.unsqueeze(-1).expand(batch_shape + (num_steps,))
    leaf_times_2 = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
    leaf_times_1 = leaf_times_2 * rate.unsqueeze(-1)

    d1 = CoalescentTimes(leaf_times_1)
    coal_times_1 = d1.sample(sample_shape)
    log_prob_1 = d1.log_prob(coal_times_1)

    d2 = CoalescentTimesWithRate(leaf_times_2, rate_grid)
    coal_times_2 = coal_times_1 / rate.unsqueeze(-1)
    log_prob_2 = d2.log_prob(coal_times_2)

    log_abs_det_jacobian = -coal_times_2.size(-1) * rate.log()
    assert_close(log_prob_1 - log_abs_det_jacobian, log_prob_2)


@pytest.mark.parametrize("clamped", [True, False], ids=["clamped", "unclamped"])
@pytest.mark.parametrize("num_steps", [2, 5, 10, 20])
@pytest.mark.parametrize("num_leaves", [2, 5, 10, 20])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
def test_likelihood_vectorized(num_leaves, num_steps, batch_shape, clamped):
    if clamped:
        leaf_times = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
        coal_times = CoalescentTimes(leaf_times).sample().clamp(min=0)
    else:
        leaf_times = torch.randn(batch_shape + (num_leaves,))
        leaf_times.mul_(0.25).add_(0.75).mul_(num_steps)
        coal_times = CoalescentTimes(leaf_times).sample()

    rate_grid = torch.rand(batch_shape + (num_steps,)) + 0.5

    d = CoalescentTimesWithRate(leaf_times, rate_grid)
    expected = d.log_prob(coal_times)

    likelihood = CoalescentRateLikelihood(leaf_times, coal_times, num_steps)
    actual = likelihood(rate_grid).sum(-1)

    assert_close(actual, expected)


@pytest.mark.parametrize("clamped", [True, False], ids=["clamped", "unclamped"])
@pytest.mark.parametrize("num_steps", [2, 5, 10, 20])
@pytest.mark.parametrize("num_leaves", [2, 5, 10, 20])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
def test_likelihood_sequential(num_leaves, num_steps, batch_shape, clamped):
    if clamped:
        leaf_times = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
        coal_times = CoalescentTimes(leaf_times).sample().clamp(min=0)
    else:
        leaf_times = torch.randn(batch_shape + (num_leaves,))
        leaf_times.mul_(0.25).add_(0.75).mul_(num_steps)
        coal_times = CoalescentTimes(leaf_times).sample()

    rate_grid = torch.rand(batch_shape + (num_steps,)) + 0.5

    d = CoalescentTimesWithRate(leaf_times, rate_grid)
    expected = d.log_prob(coal_times)

    likelihood = CoalescentRateLikelihood(leaf_times, coal_times, num_steps)
    actual = sum(likelihood(rate_grid[..., t], t)
                 for t in range(num_steps))

    assert_close(actual, expected)
