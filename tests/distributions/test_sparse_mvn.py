from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions import MultivariateNormal, SparseMultivariateNormal

from tests.common import assert_equal


def test_scale_tril():
    loc = torch.tensor([1, 2, 1, 2, 0])
    D = torch.tensor([1, 2, 3, 4, 5])
    W = torch.tensor([[1, -1, 2, 3, 4], [2, 3, 1, 2, 4]])
    cov = D.diag() + W.t().matmul(W)

    mvn = MultivariateNormal(loc, cov)
    sparse_mvn = SparseMultivariateNormal(loc, D, W)

    assert_equal(mvn.scale_tril, sparse_mvn.scale_tril)


def test_log_prob():
    loc = torch.tensor([2, 1, 1, 2, 2])
    D = torch.tensor([1, 2, 3, 1, 3])
    W = torch.tensor([[1, -1, 2, 2, 4], [2, 1, 1, 2, 6]])
    x = torch.tensor([2, 3, 4, 1, 7])
    cov = D.diag() + W.t().matmul(W)

    mvn = MultivariateNormal(loc, cov)
    sparse_mvn = SparseMultivariateNormal(loc, D, W)

    assert_equal(mvn.log_prob(x), sparse_mvn.log_prob(x))


def test_variance():
    loc = torch.tensor([1, 1, 1, 2, 0])
    D = torch.tensor([1, 2, 2, 4, 5])
    W = torch.tensor([[3, -1, 3, 3, 4], [2, 3, 1, 3, 4]])
    cov = D.diag() + W.t().matmul(W)

    mvn = MultivariateNormal(loc, cov)
    sparse_mvn = SparseMultivariateNormal(loc, D, W)

    assert_equal(mvn.variance, sparse_mvn.variance)
