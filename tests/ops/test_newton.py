# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging

import pytest
import torch
from torch.autograd import grad

from pyro.ops.newton import newton_step
from tests.common import assert_equal


logger = logging.getLogger(__name__)


def random_inside_unit_circle(shape, requires_grad=False):
    x = torch.randn(shape)
    x = x / (1 + x.pow(2).sum(-1, True))
    assert (x.pow(2).sum(-1) < 1).all()
    x = x.detach()
    if requires_grad:
        x.requires_grad = requires_grad
    return x


@pytest.mark.parametrize('batch_shape', [(), (1,), (2,), (10,), (3, 2), (2, 3)])
@pytest.mark.parametrize('trust_radius', [None, 2.0, 100.0])
@pytest.mark.parametrize('dims', [1, 2, 3])
def test_newton_step(batch_shape, trust_radius, dims):
    batch_shape = torch.Size(batch_shape)
    mode = 0.5 * random_inside_unit_circle(batch_shape + (dims,), requires_grad=True)
    x = 0.5 * random_inside_unit_circle(batch_shape + (dims,), requires_grad=True)
    if trust_radius is not None:
        assert trust_radius >= 2, '(x, mode) may be farther apart than trust_radius'

    # create a quadratic loss function
    flat_x = x.reshape(-1, dims)
    flat_mode = mode.reshape(-1, dims)
    noise = torch.randn(flat_x.shape[0], dims, 1)
    flat_hessian = noise.matmul(noise.transpose(-1, -2)) + torch.eye(dims)
    hessian = flat_hessian.reshape(batch_shape + (dims, dims))
    diff = (flat_x - flat_mode).unsqueeze(-2)
    loss = 0.5 * diff.bmm(flat_hessian).bmm(diff.transpose(-1, -2)).sum()

    # run method under test
    x_updated, cov = newton_step(loss, x, trust_radius=trust_radius)

    # check shapes
    assert x_updated.shape == x.shape
    assert cov.shape == hessian.shape

    # check values
    assert_equal(x_updated, mode, prec=1e-6,
                 msg='{} vs {}'.format(x_updated, mode))
    flat_cov = cov.reshape(flat_hessian.shape)
    assert_equal(flat_cov, flat_cov.transpose(-1, -2),
                 msg='covariance is not symmetric: {}'.format(flat_cov))
    actual_eye = torch.bmm(flat_cov, flat_hessian)
    expected_eye = torch.eye(dims).expand(actual_eye.shape)
    assert_equal(actual_eye, expected_eye, prec=1e-4,
                 msg='bad covariance {}'.format(actual_eye))

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


@pytest.mark.parametrize('trust_radius', [None, 0.1, 1.0, 10.0])
@pytest.mark.parametrize('dims', [1, 2, 3])
def test_newton_step_trust(trust_radius, dims):
    batch_size = 100
    batch_shape = torch.Size((batch_size,))
    mode = random_inside_unit_circle(batch_shape + (dims,), requires_grad=True) + 1
    x = random_inside_unit_circle(batch_shape + (dims,), requires_grad=True) - 1

    # create a quadratic loss function
    noise = torch.randn(batch_size, dims, dims)
    hessian = noise + noise.transpose(-1, -2)
    diff = (x - mode).unsqueeze(-2)
    loss = 0.5 * diff.bmm(hessian).bmm(diff.transpose(-1, -2)).sum()

    # run method under test
    x_updated, cov = newton_step(loss, x, trust_radius=trust_radius)

    # check shapes
    assert x_updated.shape == x.shape
    assert cov.shape == hessian.shape

    # check values
    if trust_radius is None:
        assert ((x - x_updated).pow(2).sum(-1) > 1.0).any(), 'test is too weak'
    else:
        assert ((x - x_updated).pow(2).sum(-1) <= 1e-8 + trust_radius**2).all(), 'trust region violated'


@pytest.mark.parametrize('trust_radius', [None, 0.1, 1.0, 10.0])
@pytest.mark.parametrize('dims', [1, 2, 3])
def test_newton_step_converges(trust_radius, dims):
    batch_size = 100
    batch_shape = torch.Size((batch_size,))
    mode = random_inside_unit_circle(batch_shape + (dims,), requires_grad=True) - 1
    x = random_inside_unit_circle(batch_shape + (dims,), requires_grad=True) + 1

    # create a quadratic loss function
    noise = torch.randn(batch_size, dims, 1)
    hessian = noise.matmul(noise.transpose(-1, -2)) + 0.01 * torch.eye(dims)

    def loss_fn(x):
        diff = (x - mode).unsqueeze(-2)
        return 0.5 * diff.bmm(hessian).bmm(diff.transpose(-1, -2)).sum()

    # check convergence
    for i in range(100):
        x = x.detach()
        x.requires_grad = True
        loss = loss_fn(x)
        x, cov = newton_step(loss, x, trust_radius=trust_radius)
        if ((x - mode).pow(2).sum(-1) < 1e-4).all():
            logger.debug('Newton iteration converged after {} steps'.format(2 + i))
            return
    pytest.fail('Newton iteration did not converge')
