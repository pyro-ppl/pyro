# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.distributions import LowRankMultivariateNormal, MultivariateNormal
from tests.common import assert_equal


def test_scale_tril():
    loc = torch.tensor([1.0, 2.0, 1.0, 2.0, 0.0])
    D = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    W = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [2.0, 1.0], [3.0, 2.0], [4.0, 4.0]])
    cov = D.diag() + W.matmul(W.t())

    mvn = MultivariateNormal(loc, cov)
    lowrank_mvn = LowRankMultivariateNormal(loc, W, D)

    assert_equal(mvn.scale_tril, lowrank_mvn.scale_tril)


def test_log_prob():
    loc = torch.tensor([2.0, 1.0, 1.0, 2.0, 2.0])
    D = torch.tensor([1.0, 2.0, 3.0, 1.0, 3.0])
    W = torch.tensor([[1.0, 2.0], [-1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [4.0, 6.0]])
    x = torch.tensor([2.0, 3.0, 4.0, 1.0, 7.0])
    cov = D.diag() + W.matmul(W.t())

    mvn = MultivariateNormal(loc, cov)
    lowrank_mvn = LowRankMultivariateNormal(loc, W, D)

    assert_equal(mvn.log_prob(x), lowrank_mvn.log_prob(x))


def test_variance():
    loc = torch.tensor([1.0, 1.0, 1.0, 2.0, 0.0])
    D = torch.tensor([1.0, 2.0, 2.0, 4.0, 5.0])
    W = torch.tensor([[3.0, 2.0], [-1.0, 3.0], [3.0, 1.0], [3.0, 3.0], [4.0, 4.0]])
    cov = D.diag() + W.matmul(W.t())

    mvn = MultivariateNormal(loc, cov)
    lowrank_mvn = LowRankMultivariateNormal(loc, W, D)

    assert_equal(mvn.variance, lowrank_mvn.variance)
