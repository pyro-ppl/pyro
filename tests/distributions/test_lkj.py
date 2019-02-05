from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.distributions import biject_to, transform_to

from pyro.distributions.lkj import (corr_cholesky_constraint, _TanhTransform,
                                    _PartialCorrToCorrCholeskyTransform)
from tests.common import assert_tensors_equal


@pytest.mark.parametrize("value_shape", [(1, 1), (3, 1, 1), (3, 3), (1, 3, 3), (5, 3, 3)])
def test_constraint(value_shape):
    value = torch.randn(value_shape).tril()
    value.diagonal(dim1=-2, dim2=-1).exp_()
    value = value / value.norm(2, dim=-1, keepdim=True)

    # this also tests for shape
    assert_tensors_equal(corr_cholesky_constraint.check(value), torch.ones(value_shape[:-2]))


@pytest.mark.parametrize("x_shape", [(), (1,), (3, 1), (3,), (1, 3), (5, 3)])
def test_tanh_transform(x_shape):
    transform = _TanhTransform()
    x = torch.randn(x_shape, requires_grad=True)
    y = transform(x)

    # test codomain
    assert_tensors_equal(transform.codomain.check(y), torch.ones(x_shape))

    # test inv
    z = transform.inv(y)
    assert_tensors_equal(x, z)

    # test log_abs_det_jacobian
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det.shape == x_shape
    if x_shape == ():
        assert_tensors_equal(torch.autograd.grad(y, (x,), retain_graph=True)[0].abs().log(), log_det)
        assert_tensors_equal(torch.autograd.grad(z, (y,), retain_graph=True)[0].abs().log(), -log_det)


def _autograd_log_det(ys, x):
    # computes log_abs_det_jacobian of y w.r.t. x
    return torch.stack([torch.autograd.grad(y, (x,), retain_graph=True)[0]
                        for y in ys]).det().abs().log()


@pytest.mark.parametrize("x_shape", [(1,), (3, 1), (6,), (1, 6), (5, 6)])
def test_partial_corr_to_corr_cholesky_transform(x_shape):
    transform = _PartialCorrToCorrCholeskyTransform()
    x = torch.empty(x_shape).uniform_(-1, 1).requires_grad_()
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
    if len(x_shape) == 1:
        triu_index = y.new_ones(y.shape).triu(diagonal=1) > 0.5
        y_tril_vector = y.t()[triu_index]
        assert_tensors_equal(_autograd_log_det(y_tril_vector, x), log_det)

        y_tril_vector = y_tril_vector.detach().requires_grad_()
        y = y.new_zeros(y.shape)
        y[triu_index] = y_tril_vector
        y = y.t()
        z = transform.inv(y)
        assert_tensors_equal(_autograd_log_det(z, y_tril_vector), -log_det)


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
    if len(x_shape) == 1:
        triu_index = y.new_ones(y.shape).triu(diagonal=1) > 0.5
        y_tril_vector = y.t()[triu_index]
        assert_tensors_equal(_autograd_log_det(y_tril_vector, x), log_det)

        y_tril_vector = y_tril_vector.detach().requires_grad_()
        y = y.new_zeros(y.shape)
        y[triu_index] = y_tril_vector
        y = y.t()
        z = transform.inv(y)
        assert_tensors_equal(_autograd_log_det(z, y_tril_vector), -log_det)
