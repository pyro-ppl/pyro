from collections import OrderedDict

import pytest
import torch

from pyro.ops.welford import WelfordCovariance
from tests.common import assert_equal


@pytest.mark.parametrize('n_samples,dim_size', [(10000, 1),
                                                (10000, 7),
                                                pytest.mark.xfail((1, 10))])  # Insufficient samples
@pytest.mark.init(rng_seed=7)
def test_welford_diagonal(n_samples, dim_size):
    w = WelfordCovariance(diagonal=True)
    loc = torch.zeros(dim_size)
    cov_diagonal = torch.rand(dim_size)
    cov = torch.diag(cov_diagonal)
    dist = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    for _ in range(n_samples):
        sample = OrderedDict((i, v) for i, v in enumerate(dist.sample()))
        w.update(sample)
    estimates = w.get_estimates(regularize=False).squeeze(-1)
    # Since Welford is in estimation, allow for a reasonable margin of error
    assert_equal(cov_diagonal, estimates, prec=1e-1)


@pytest.mark.parametrize('n_samples,dim_size', [(20000, 1),
                                                (20000, 7),
                                                pytest.mark.xfail((1, 10))])
@pytest.mark.init(rng_seed=7)
def test_welford_dense(n_samples, dim_size):
    w = WelfordCovariance(diagonal=False)
    loc = torch.zeros(dim_size)
    cov = torch.randn(dim_size, dim_size)
    cov = torch.mm(cov, cov.t())
    dist = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    for _ in range(n_samples):
        sample = OrderedDict((i, v) for i, v in enumerate(dist.sample()))
        w.update(sample)
    estimates = w.get_estimates(regularize=False)
    assert_equal(cov, estimates, prec=3e-1)
