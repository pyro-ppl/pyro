import pytest
import torch

from pyro.distributions.empirical import Empirical
from tests.common import assert_equal


@pytest.mark.parametrize("size", [torch.Size(), torch.Size((1,)), torch.Size((2, 3))])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_mean_and_var(size, dtype):
    empirical_dist = Empirical()
    for i in range(5):
        empirical_dist.add(torch.ones(size, dtype=dtype) * i)
    true_mean = torch.ones(size) * 2
    true_var = torch.ones(size) * 2.5
    assert_equal(empirical_dist.mean, true_mean)
    assert_equal(empirical_dist.variance, true_var)


@pytest.mark.parametrize("batch_shape", [torch.Size(), torch.Size((1,)), torch.Size((2, 3))])
@pytest.mark.parametrize("sample_shape", [torch.Size((20,)), torch.Size((20, 3, 4))])
@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.float64])
def test_sample(batch_shape, sample_shape, dtype):
    empirical_dist = Empirical()
    for i in range(5):
        empirical_dist.add(torch.ones(batch_shape, dtype=dtype) * i)
    samples = empirical_dist.sample(sample_shape=sample_shape)
    assert_equal(samples.size(), sample_shape + batch_shape)
    assert_equal(set(samples.view(-1).tolist()), set(range(5)))
