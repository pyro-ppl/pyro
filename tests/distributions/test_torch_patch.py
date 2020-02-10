# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_close, requires_cuda


@requires_cuda
def test_dirichlet_grad_cuda():
    concentration = torch.ones(3, requires_grad=True)
    dist.Dirichlet(concentration).rsample().sum().backward()


@requires_cuda
def test_linspace():
    x = torch.linspace(-1., 1., 100, device="cuda")
    assert x.device.type == "cuda"


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_lower_cholesky_transform(batch_shape, dim):
    t = torch.distributions.transform_to(torch.distributions.constraints.lower_cholesky)
    x = torch.randn(batch_shape + (dim, dim))
    y = t(x)
    assert y.shape == x.shape
    actual = y.matmul(y.transpose(-1, -2)).cholesky()
    assert_close(actual, y)
    x2 = t.inv(y)
    assert x2.shape == x.shape
    y2 = t(x2)
    assert_close(y2, y)
