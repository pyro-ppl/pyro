import numpy as np
import pytest
import torch

from pyro.ops.welford import WelfordCovariance
from tests.common import assert_equal


@pytest.mark.parametrize('n_samples,dim_size', [(1000, 1),
                                                (1000, 7),
                                                pytest.mark.xfail((1, 1))])  # Insufficient samples
@pytest.mark.init(rng_seed=7)
def test_welford_diagonal(n_samples, dim_size):
    w = WelfordCovariance(diagonal=True)
    loc = torch.zeros(dim_size)
    cov_diagonal = torch.rand(dim_size)
    cov = torch.diag(cov_diagonal)
    dist = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    samples = []
    for _ in range(n_samples):
        sample = dist.sample()
        samples.append(sample)
        w.update(sample)

    sample_variance = torch.stack(samples).var(dim=0, unbiased=True)
    estimates = w.get_covariance(regularize=False)
    assert_equal(estimates, sample_variance)


@pytest.mark.parametrize('n_samples,dim_size', [(1000, 1),
                                                (1000, 7),
                                                pytest.mark.xfail((1, 1))])
@pytest.mark.init(rng_seed=7)
def test_welford_dense(n_samples, dim_size):
    w = WelfordCovariance(diagonal=False)
    loc = torch.zeros(dim_size)
    cov = torch.randn(dim_size, dim_size)
    cov = torch.mm(cov, cov.t())
    dist = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    samples = []
    for _ in range(n_samples):
        sample = dist.sample()
        samples.append(sample)
        w.update(sample)

    sample_cov = np.cov(torch.stack(samples).data.cpu().numpy(), bias=False, rowvar=False)
    estimates = w.get_covariance(regularize=False).data.cpu().numpy()
    assert_equal(estimates, sample_cov)
