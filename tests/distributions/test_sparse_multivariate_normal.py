from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions import MultivariateNormal, SparseMultivariateNormal

from tests.common import assert_equal


def test_scaleu():
    loc = Variable(torch.Tensor([1, 2, 1, 2, 0]))
    D = Variable(torch.Tensor([1, 2, 3, 4, 5]))
    W = Variable(torch.Tensor([[1, -1, 2, 3, 4], [2, 3, 1, 2, 4]]))
    cov = D + W.t().matmul(W)

    mvn = MultivariateNormal(loc, cov)
    sparse_mvn = SparseMultivariateNormal(loc, D, W)

    assert_equal(mvn.scaleu, sparse_mvn.scaleu)


def test_batch_log_pdf():
    loc = Variable(torch.Tensor([2, 1, 1, 2, 2]))
    D = Variable(torch.Tensor([1, 2, 3, 1, 3]))
    W = Variable(torch.Tensor([[1, -1, 2, 2, 4], [2, 1, 1, 2, 6]]))
    y = Variable(torch.Tensor([2, 3, 4, 1, 7]))
    cov = D + W.t().matmul(W)

    mvn = MultivariateNormal(loc, cov)
    sparse_mvn = SparseMultivariateNormal(loc, D, W)

    assert_equal(mvn.batch_log_pdf(y), sparse_mvn.batch_log_pdf(y))
