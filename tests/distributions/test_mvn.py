# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.distributions import MultivariateNormal
from tests.common import assert_equal


def random_mvn(loc_shape, cov_shape, dim):
    """
    Generate a random MultivariateNormal distribution for testing.
    """
    rank = dim + dim
    loc = torch.randn(loc_shape + (dim,))
    cov = torch.randn(cov_shape + (dim, rank))
    cov = cov.matmul(cov.transpose(-1, -2))
    return MultivariateNormal(loc, cov)


@pytest.mark.parametrize('loc_shape', [
    (), (2,), (3, 2),
])
@pytest.mark.parametrize('cov_shape', [
    (), (2,), (3, 2),
])
@pytest.mark.parametrize('dim', [
    1, 3, 5,
])
def test_shape(loc_shape, cov_shape, dim):
    mvn = random_mvn(loc_shape, cov_shape, dim)
    mvn._unbroadcasted_scale_tril.requires_grad_()
    assert mvn.loc.shape == mvn.batch_shape + mvn.event_shape
    assert mvn.covariance_matrix.shape == mvn.batch_shape + mvn.event_shape * 2
    assert mvn.scale_tril.shape == mvn.covariance_matrix.shape
    assert mvn.precision_matrix.shape == mvn.covariance_matrix.shape

    assert_equal(mvn.precision_matrix, mvn.covariance_matrix.inverse())

    # smoke test for precision backward
    mvn.precision_matrix.sum().backward()
