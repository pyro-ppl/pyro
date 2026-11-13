# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from pyro.distributions import SoftLaplace


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_icdf_endpoints(dtype):
    d = SoftLaplace(torch.tensor(2.0, dtype=dtype), torch.tensor(3.0, dtype=dtype))
    actual = d.icdf(torch.tensor([0.0, 1.0], dtype=dtype))
    assert torch.equal(actual, torch.tensor([-math.inf, math.inf], dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_icdf_upper_tail_symmetry(dtype):
    d = SoftLaplace(torch.tensor(2.0, dtype=dtype), torch.tensor(3.0, dtype=dtype))
    p = torch.tensor([0.75, 0.99, 1.0 - torch.finfo(dtype).eps], dtype=dtype)
    actual = d.icdf(p)
    reflected = 2 * d.loc - d.icdf(1 - p)
    torch.testing.assert_close(actual, reflected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_icdf_interior_gradient(dtype):
    d = SoftLaplace(torch.tensor(2.0, dtype=dtype), torch.tensor(3.0, dtype=dtype))
    p = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=dtype, requires_grad=True)
    value = d.icdf(p)
    (gradient,) = torch.autograd.grad(value.sum(), (p,))
    expected = math.pi * d.scale / (math.pi * p).sin()
    torch.testing.assert_close(gradient, expected)
    torch.testing.assert_close(d.cdf(value), p)
