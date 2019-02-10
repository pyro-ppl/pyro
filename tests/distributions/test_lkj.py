from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.distributions import biject_to, transform_to
from pyro.distributions.lkj import (corr_cholesky_constraint, CorrLCholeskyTransform)
from tests.common import assert_tensors_equal


@pytest.mark.parametrize("value_shape", [(1, 1), (3, 3), (5, 5)])
def test_constraint(value_shape):
    value = torch.randn(value_shape).clamp(-6, 6).tril()
    value.diagonal(dim1=-2, dim2=-1).exp_()
    value = value / value.norm(2, dim=-1, keepdim=True)

    #assert_tensors_equal(corr_cholesky_constraint.check(value), torch.ones(value_shape[:-2], dtype=value.dtype))
    assert (corr_cholesky_constraint.check(value) == 1).all()


def _autograd_log_det(ys, x):
    # computes log_abs_det_jacobian of y w.r.t. x
    return torch.stack([torch.autograd.grad(y, (x,), retain_graph=True)[0]
                        for y in ys]).det().abs().log()


@pytest.mark.parametrize("y_shape", [(1,), (3, 1), (6,), (1, 6), (2, 6)])
def test_unconstrained_to_corr_cholesky_transform(y_shape):
    transform = CorrLCholeskyTransform()
    y = torch.empty(y_shape).normal_(0, 4).clamp(-6, 6).requires_grad_()
    x = transform(y)

    # test codomain
    assert (transform.codomain.check(x) == 1).all()

    # test inv
    y_prime = transform.inv(x)
    assert_tensors_equal(y, y_prime, prec=1e-4)

    # test domain
    assert (transform.domain.check(y_prime) == 1).all()

    # test log_abs_det_jacobian
    log_det = transform.log_abs_det_jacobian(y, x)
    assert log_det.shape == y_shape[:-1]


@pytest.mark.parametrize("x_shape", [(1,), (3, 1), (6,), (1, 6), (5, 6)])
@pytest.mark.parametrize("mapping", [biject_to, transform_to])
def test_corr_cholesky_transform(x_shape, mapping):
    transform = mapping(corr_cholesky_constraint)
    x = torch.randn(x_shape, requires_grad=True).clamp(-6, 6)
    y = transform(x)

    # test codomain
    assert (transform.codomain.check(y) == 1).all()

    # test inv
    z = transform.inv(y)
    assert_tensors_equal(x, z, prec=1e-4)

    # test domain
    assert (transform.domain.check(z) == 1).all()

    # test log_abs_det_jacobian
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det.shape == x_shape[:-1]
