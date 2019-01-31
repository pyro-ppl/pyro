from __future__ import absolute_import, division, print_function

import pytest
import torch

import pyro.distributions as dist
from pyro.distributions.lkj import corr_cholesky_constraint, _TanhTransform
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
        assert_tensors_equal(torch.autograd.grad(y, (x,), retain_graph=True)[0].abs().log(),
                             log_det, prec=0.0002)
        assert_tensors_equal(torch.autograd.grad(z, (y,), retain_graph=True)[0].abs().log(),
                             -log_det, prec=0.0002)


def test_partial_corr_to_corr_cholesky_transform():
    pass


def test_biject_to():
    pass


def test_transform_to():
    pass
