# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pyro.ops.welford import WelfordArrowheadCovariance, WelfordCovariance
from pyro.util import optional
from tests.common import assert_equal


@pytest.mark.parametrize('n_samples,dim_size', [(1000, 1),
                                                (1000, 7),
                                                (1, 1)])
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
    with optional(pytest.raises(RuntimeError), n_samples == 1):
        estimates = w.get_covariance(regularize=False)
        assert_equal(estimates, sample_variance)


@pytest.mark.parametrize('n_samples,dim_size', [(1000, 1),
                                                (1000, 7),
                                                (1, 1)])
@pytest.mark.init(rng_seed=7)
def test_welford_dense(n_samples, dim_size):
    w = WelfordCovariance(diagonal=False)
    loc = torch.zeros(dim_size)
    cov = torch.randn(dim_size, dim_size)
    cov = torch.mm(cov, cov.t())
    dist = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    samples = dist.sample(torch.Size([n_samples]))
    for sample in samples:
        w.update(sample)

    with optional(pytest.raises(RuntimeError), n_samples == 1):
        estimates = w.get_covariance(regularize=False).cpu().numpy()
        sample_cov = np.cov(samples.cpu().numpy(), bias=False, rowvar=False)
        assert_equal(estimates, sample_cov)


@pytest.mark.parametrize('n_samples,dim_size,head_size', [
    (1000, 5, 0),
    (1000, 5, 1),
    (1000, 5, 4),
    (1000, 5, 5)
])
@pytest.mark.parametrize('regularize', [True, False])
def test_welford_arrowhead(n_samples, dim_size, head_size, regularize):
    adapt_scheme = WelfordArrowheadCovariance(head_size=head_size)
    loc = torch.zeros(dim_size)
    cov = torch.randn(dim_size, dim_size)
    cov = torch.mm(cov, cov.t())
    dist = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    samples = dist.sample(sample_shape=torch.Size([n_samples]))

    for sample in samples:
        adapt_scheme.update(sample)
    top, bottom_diag = adapt_scheme.get_covariance(regularize=regularize)
    actual = torch.cat([top, torch.cat([top[:, head_size:].t(), bottom_diag.diag()], -1)])

    mask = torch.ones(dim_size, dim_size)
    mask[head_size:, head_size:] = 0.
    mask.view(-1)[::dim_size + 1][head_size:] = 1.
    expected = np.cov(samples.cpu().numpy(), bias=False, rowvar=False)
    expected = torch.from_numpy(expected).type_as(mask)
    if regularize:
        expected = (expected * n_samples + 1e-3 * torch.eye(dim_size) * 5) / (n_samples + 5)
    expected = expected * mask
    assert_equal(actual, expected)
