from __future__ import absolute_import, division, print_function

import itertools

import pytest
import torch
from torch.autograd import grad

from pyro.ops.newton import newton_step_2d
from tests.common import assert_equal, xfail_param


def random_inside_unit_circle(shape, requires_grad=False):
    x = torch.randn(shape)
    x = x / (1 + x.pow(2).sum(-1, True))
    assert (x.pow(2).sum(-1) < 1).all()
    x = x.detach()
    if requires_grad:
        x.requires_grad = requires_grad
    return x


@pytest.mark.parametrize('trust_radius', [None, 2.0, 100.0])
@pytest.mark.parametrize('batch_shape', [
    (),
    (1,),
    xfail_param((4,)),
    xfail_param((3, 2)),
])
def test_quadratic_near_optimum(batch_shape, trust_radius):
    batch_shape = torch.Size(batch_shape)
    mode = random_inside_unit_circle(batch_shape + (2,), requires_grad=True)
    x = random_inside_unit_circle(batch_shape + (2,), requires_grad=True)
    if trust_radius is not None:
        assert trust_radius >= 2, 'x, mode may be farther apart than trust_radius'

    flat_x = x.reshape(-1, 2)
    flat_mode = mode.reshape(-1, 2)
    flat_hessian_sqrt = torch.randn(flat_x.shape[0], 2, 2)
    flat_hessian = flat_hessian_sqrt.bmm(flat_hessian_sqrt.transpose(-1, -2))
    hessian = flat_hessian.reshape(batch_shape + (2, 2))
    loss = 0.5 * (flat_x - flat_mode).matmul(flat_hessian_sqrt).pow(2).sum()

    # run method under test
    x_updated, cov = newton_step_2d(loss, x, trust_radius=trust_radius)

    # check shapes
    assert x_updated.shape == x.shape
    assert cov.shape == hessian.shape

    # check values
    assert_equal(x_updated, mode, prec=1e-6,
                 msg='{} vs {}'.format(x_updated, mode))
    flat_cov = cov.reshape(flat_hessian.shape)
    assert_equal(flat_cov, flat_cov.transpose(-1, -2), msg=str(flat_cov))
    actual_eye = torch.bmm(flat_cov, flat_hessian)
    expected_eye = torch.eye(2, 2).expand(actual_eye.shape)
    assert_equal(actual_eye, expected_eye, prec=1e-4, msg='{}'.format(actual_eye))

    # check gradients
    for i in itertools.product(*map(range, mode.shape)):
        expected_grad = torch.zeros(mode.shape)
        expected_grad[i] = 1
        actual_grad = grad(x_updated[i], [mode], create_graph=True)[0]
        assert_equal(actual_grad, expected_grad, prec=1e-5, msg='\n'.join([
            'bad gradient at index {}'.format(i),
            'expected {}'.format(expected_grad),
            'actual   {}'.format(actual_grad),
        ]))
