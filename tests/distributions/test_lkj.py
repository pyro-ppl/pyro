from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.distributions import biject_to, transform_to
from pyro.distributions.lkj import (corr_cholesky_constraint, UnconstrainedToCorrLCholeskyTransform,
                                    _PartialCorrToCorrLCholeskyTransform)
from tests.common import assert_tensors_equal


@pytest.mark.parametrize("value_shape", [(1, 1), (3, 3), (5, 5)])
def test_constraint(value_shape):
    value = torch.randn(value_shape).tril()
    value.diagonal(dim1=-2, dim2=-1).exp_()
    value = value / value.norm(2, dim=-1, keepdim=True)

    # this also tests for shape
    assert_tensors_equal(corr_cholesky_constraint.check(value), torch.ones(value_shape[:-2]))


def _autograd_log_det(ys, x):
    # computes log_abs_det_jacobian of y w.r.t. x
    return torch.stack([torch.autograd.grad(y, (x,), retain_graph=True)[0]
                        for y in ys]).det().abs().log()


@pytest.mark.parametrize("x_shape", [(1,), (3, 1), (6,), (1, 6), (5, 6)])
def test_partial_corr_to_corr_cholesky_transform(x_shape):
    transform = _PartialCorrToCorrLCholeskyTransform()
    y = torch.empty(x_shape).uniform_(-1, 1).requires_grad_()
    x = transform(y)

    # test codomain
    assert_tensors_equal(transform.codomain.check(x), torch.ones(y.shape[:-1]))

    # test inv
    y_prime = transform.inv(x)
    assert_tensors_equal(y, y_prime)

    # test domain
    assert_tensors_equal(transform.domain.check(y_prime), torch.ones(x_shape))

    # test log_abs_det_jacobian
    log_det = transform.log_abs_det_jacobian(y, x)
    assert log_det.shape == x_shape[:-1]


@pytest.mark.parametrize("x_shape", [(1,), (3, 1), (6,), (1, 6), (5, 6)])
@pytest.mark.parametrize("mapping", [biject_to, transform_to])
def test_corr_cholesky_transform(x_shape, mapping):
    transform = mapping(corr_cholesky_constraint)
    x = torch.randn(x_shape, requires_grad=True)
    y = transform(x)

    # test codomain
    assert_tensors_equal(transform.codomain.check(y), torch.ones(x.shape[:-1]))

    # test inv
    z = transform.inv(y)
    assert_tensors_equal(x, z)

    # test domain
    assert_tensors_equal(transform.domain.check(z), torch.ones(x_shape))

    # test log_abs_det_jacobian
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det.shape == x_shape[:-1]
