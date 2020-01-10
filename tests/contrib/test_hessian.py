# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro.distributions as dist
from pyro.ops.hessian import hessian
from tests.common import assert_equal


def test_hessian_mvn():
    tmp = torch.randn(3, 10)
    cov = torch.matmul(tmp, tmp.t())
    mvn = dist.MultivariateNormal(cov.new_zeros(3), cov)

    x = torch.randn(3, requires_grad=True)
    y = mvn.log_prob(x)
    assert_equal(hessian(y, x), -mvn.precision_matrix)


def test_hessian_multi_variables():
    x = torch.randn(3, requires_grad=True)
    z = torch.randn(3, requires_grad=True)
    y = (x ** 2 * z + z ** 3).sum()

    H = hessian(y, (x, z))
    Hxx = (2 * z).diag()
    Hxz = (2 * x).diag()
    Hzz = (6 * z).diag()
    target_H = torch.cat([torch.cat([Hxx, Hxz]), torch.cat([Hxz, Hzz])], dim=1)
    assert_equal(H, target_H)
