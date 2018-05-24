import pytest
import torch

from pyro.distributions.empirical import Empirical
from tests.common import assert_equal


@pytest.mark.parametrize("size", [torch.Size(), torch.Size((1,)), torch.Size((2, 3))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unweighted_mean_and_var(size, dtype):
    empirical_dist = Empirical()
    for i in range(5):
        empirical_dist.add(torch.ones(size, dtype=dtype) * i)
    true_mean = torch.ones(size) * 2
    true_var = torch.ones(size) * 2
    assert_equal(empirical_dist.mean, true_mean)
    assert_equal(empirical_dist.variance, true_var)


@pytest.mark.parametrize("batch_shape", [torch.Size(), torch.Size((1,)), torch.Size((2, 3))])
@pytest.mark.parametrize("sample_shape", [torch.Size((20,)), torch.Size((20, 3, 4))])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_unweighted_samples(batch_shape, sample_shape, dtype):
    empirical_dist = Empirical()
    for i in range(5):
        empirical_dist.add(torch.ones(batch_shape, dtype=dtype) * i)
    samples = empirical_dist.sample(sample_shape=sample_shape)
    assert_equal(samples.size(), sample_shape + batch_shape)
    assert_equal(set(samples.view(-1).tolist()), set(range(5)))


@pytest.mark.parametrize("batch_shape", [torch.Size(), torch.Size((1,)), torch.Size((2, 3))])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_log_prob(batch_shape, dtype):
    empirical_dist = Empirical()
    for i in range(5):
        empirical_dist.add(torch.ones(batch_shape, dtype=dtype) * i)
    sample_to_score = torch.ones(batch_shape, dtype=dtype)
    log_prob = empirical_dist.log_prob(sample_to_score)
    assert_equal(log_prob, torch.tensor(0.2).log())

    # Value outside support returns -Inf
    sample_to_score = torch.ones(batch_shape, dtype=dtype) * 6
    log_prob = empirical_dist.log_prob(sample_to_score)
    assert log_prob == -float("inf")

    # Vectorized ``log_prob`` raises ValueError
    with pytest.raises(ValueError):
        sample_to_score = torch.ones((3,) + batch_shape, dtype=dtype)
        empirical_dist.log_prob(sample_to_score)


@pytest.mark.parametrize("event_shape", [torch.Size(), torch.Size((1,)), torch.Size((2, 3))])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_weighted_sample_coherence(event_shape, dtype):
    samples = [(1.0, 0.5), (0.0, 1.5), (1.0, 0.5), (0.0, 1.5)]
    empirical_dist = Empirical()
    for sample, weight in samples:
        empirical_dist.add(sample * torch.ones(event_shape, dtype=dtype), weight=weight)
    assert_equal(empirical_dist.event_shape, event_shape)
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


@pytest.mark.parametrize("event_shape", [torch.Size(), torch.Size((1,)), torch.Size((2, 3))])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_weighted_mean_var(event_shape, dtype):
    samples = [(1.0, 0.5), (0.0, 1.5), (1.0, 0.5), (0.0, 1.5)]
    empirical_dist = Empirical()
    for sample, weight in samples:
        empirical_dist.add(sample * torch.ones(event_shape, dtype=dtype), weight=weight)
    if dtype in (torch.float32, torch.float64):
        true_mean = torch.ones(event_shape, dtype=dtype) * 0.25
        true_var = torch.ones(event_shape, dtype=dtype) * 0.1875
        assert_equal(empirical_dist.mean, true_mean)
        assert_equal(empirical_dist.variance, true_var)
    else:
        with pytest.raises(ValueError):
            empirical_dist.mean
            empirical_dist.variance
