# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.distributions.empirical import Empirical
from tests.common import assert_equal, assert_close


@pytest.mark.parametrize("size", [[], [1], [2, 3]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unweighted_mean_and_var(size, dtype):
    samples = []
    for i in range(5):
        samples.append(torch.ones(size, dtype=dtype) * i)
    samples = torch.stack(samples)
    empirical_dist = Empirical(samples, torch.ones(5, dtype=dtype))
    true_mean = torch.ones(size) * 2
    true_var = torch.ones(size) * 2
    assert_equal(empirical_dist.mean, true_mean)
    assert_equal(empirical_dist.variance, true_var)


@pytest.mark.parametrize("batch_shape, event_shape", [
    ([], []),
    ([2], []),
    ([2], [5]),
    ([2], [5, 3]),
    ([2, 5], [3]),
])
@pytest.mark.parametrize("sample_shape", [[], [20], [20, 3, 4]])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_unweighted_samples(batch_shape, event_shape, sample_shape, dtype):
    agg_dim_size = 5
    # empirical samples with desired shape
    dim_ordering = list(range(len(batch_shape + event_shape) + 1))  # +1 for agg dim
    dim_ordering.insert(len(batch_shape), dim_ordering.pop())
    emp_samples = torch.arange(agg_dim_size, dtype=dtype)\
        .expand(batch_shape + event_shape + [agg_dim_size])\
        .permute(dim_ordering)
    # initial weight assignment
    weights = torch.ones(batch_shape + [agg_dim_size])
    empirical_dist = Empirical(emp_samples, weights)
    samples = empirical_dist.sample(sample_shape=torch.Size(sample_shape))
    assert_equal(samples.size(), torch.Size(sample_shape + batch_shape + event_shape))


@pytest.mark.parametrize("sample, weights, expected_mean, expected_var", [(
        torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
        torch.ones(2),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor([0.25, 0.25, 0.25]),
     ), (
        torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
        torch.ones(2, 3),
        torch.tensor([0., 1.]),
        torch.tensor([0., 0.]),
    ),
])
def test_sample_examples(sample, weights, expected_mean, expected_var):
    emp_dist = Empirical(sample, weights)
    num_samples = 10000
    assert_equal(emp_dist.mean, expected_mean)
    assert_equal(emp_dist.variance, expected_var)
    emp_samples = emp_dist.sample((num_samples,))
    assert_close(emp_samples.mean(0), emp_dist.mean, rtol=1e-2)
    assert_close(emp_samples.var(0), emp_dist.variance, rtol=1e-2)


@pytest.mark.parametrize("batch_shape, event_shape", [
    ([], []),
    ([1], []),
    ([10], []),
    ([10, 8], [3]),
    ([10, 8], [3, 4]),
])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_log_prob(batch_shape, event_shape, dtype):
    samples = []
    for i in range(5):
        samples.append(torch.ones(event_shape, dtype=dtype) * i)
    samples = torch.stack(samples).expand(batch_shape + [5] + event_shape)
    weights = torch.tensor(1.).expand(batch_shape + [5])
    empirical_dist = Empirical(samples, weights)
    sample_to_score = torch.tensor(1, dtype=dtype).expand(batch_shape + event_shape)
    log_prob = empirical_dist.log_prob(sample_to_score)
    assert_equal(log_prob, (weights.new_ones(batch_shape + [1]) * 0.2).sum(-1).log())

    # Value outside support returns -Inf
    sample_to_score = torch.tensor(1, dtype=dtype).expand(batch_shape + event_shape) * 6
    log_prob = empirical_dist.log_prob(sample_to_score)
    assert log_prob.shape == torch.Size(batch_shape)
    assert torch.isinf(log_prob).all()

    # Vectorized ``log_prob`` raises ValueError
    with pytest.raises(ValueError):
        sample_to_score = torch.ones([3] + batch_shape + event_shape, dtype=dtype)
        empirical_dist.log_prob(sample_to_score)


@pytest.mark.parametrize("event_shape", [[], [1], [2, 3]])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_weighted_sample_coherence(event_shape, dtype):
    data = [(1.0, 0.5), (0.0, 1.5), (1.0, 0.5), (0.0, 1.5)]
    samples, weights = [], []
    for sample, weight in data:
        samples.append(sample * torch.ones(event_shape, dtype=dtype))
        weights.append(torch.tensor(weight).log())
    samples, weights = torch.stack(samples), torch.stack(weights)
    empirical_dist = Empirical(samples, weights)
    assert_equal(empirical_dist.event_shape, torch.Size(event_shape))
    assert_equal(empirical_dist.sample_size, 4)
    sample_to_score = torch.ones(event_shape, dtype=dtype) * 1.0
    assert_equal(empirical_dist.log_prob(sample_to_score), torch.tensor(0.25).log())
    samples = empirical_dist.sample(sample_shape=torch.Size((1000,)))
    zeros = torch.zeros(event_shape, dtype=dtype)
    ones = torch.ones(event_shape, dtype=dtype)
    num_zeros = samples.eq(zeros).contiguous().view(1000, -1).min(dim=-1)[0].float().sum()
    num_ones = samples.eq(ones).contiguous().view(1000, -1).min(dim=-1)[0].float().sum()
    assert_equal(num_zeros.item() / 1000, 0.75, prec=0.02)
    assert_equal(num_ones.item() / 1000, 0.25, prec=0.02)


@pytest.mark.parametrize("batch_shape", [[], [1], [2], [2, 3]])
@pytest.mark.parametrize("event_shape", [[], [1], [2, 3]])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_weighted_mean_var(event_shape, dtype, batch_shape):
    data = [(1, 0.5), (0, 1.5), (1, 0.5), (0, 1.5)]
    samples, weights = [], []
    for sample, weight in data:
        samples.append(sample * torch.ones(event_shape, dtype=dtype))
        weight_dtype = dtype if dtype is not torch.long else None
        weights.append(torch.tensor(weight, dtype=weight_dtype).log())
    samples = torch.stack(samples).expand(batch_shape + [4] + event_shape)
    weights = torch.stack(weights).expand(batch_shape + [4])
    empirical_dist = Empirical(samples, weights)
    if dtype in (torch.float32, torch.float64):
        true_mean = torch.ones(batch_shape + event_shape, dtype=dtype) * 0.25
        true_var = torch.ones(batch_shape + event_shape, dtype=dtype) * 0.1875
        assert_equal(empirical_dist.mean, true_mean)
        assert_equal(empirical_dist.variance, true_var)
    else:
        with pytest.raises(ValueError):
            empirical_dist.mean
            empirical_dist.variance


def test_mean_var_non_nan():
    true_mean = torch.randn([1, 2, 3])
    samples, weights = [], []
    for i in range(10):
        samples.append(true_mean)
        weights.append(torch.tensor(-1000.))
    samples, weights = torch.stack(samples), torch.stack(weights)
    empirical_dist = Empirical(samples, weights)
    assert_equal(empirical_dist.mean, true_mean)
    assert_equal(empirical_dist.variance, torch.zeros_like(true_mean))
