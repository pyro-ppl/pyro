# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.distributions import AffineTransform, Beta, TransformedDistribution, biject_to, transform_to

from pyro.distributions import constraints, transforms
from pyro.distributions.lkj import LKJCorrCholesky
from tests.common import assert_equal, assert_tensors_equal


@pytest.mark.parametrize("value_shape", [(1, 1), (3, 3), (5, 5)])
def test_constraint(value_shape):
    value = torch.randn(value_shape).clamp(-2, 2).tril()
    value.diagonal(dim1=-2, dim2=-1).exp_()
    value = value / value.norm(2, dim=-1, keepdim=True)

    assert (constraints.corr_cholesky_constraint.check(value) == 1).all()


def _autograd_log_det(ys, x):
    # computes log_abs_det_jacobian of y w.r.t. x
    return torch.stack([torch.autograd.grad(y, (x,), retain_graph=True)[0]
                        for y in ys]).det().abs().log()


@pytest.mark.parametrize("y_shape", [(1,), (3, 1), (6,), (1, 6), (2, 6)])
def test_unconstrained_to_corr_cholesky_transform(y_shape):
    transform = transforms.CorrLCholeskyTransform()
    y = torch.empty(y_shape).uniform_(-4, 4).requires_grad_()
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
    if len(y_shape) == 1:
        triu_index = x.new_ones(x.shape).triu(diagonal=1).to(torch.bool)
        x_tril_vector = x.t()[triu_index]
        assert_tensors_equal(_autograd_log_det(x_tril_vector, y), log_det, prec=1e-4)

        x_tril_vector = x_tril_vector.detach().requires_grad_()
        x = x.new_zeros(x.shape)
        x[triu_index] = x_tril_vector
        x = x.t()
        z = transform.inv(x)
        assert_tensors_equal(_autograd_log_det(z, x_tril_vector), -log_det, prec=1e-4)


@pytest.mark.parametrize("x_shape", [(1,), (3, 1), (6,), (1, 6), (5, 6)])
@pytest.mark.parametrize("mapping", [biject_to, transform_to])
def test_corr_cholesky_transform(x_shape, mapping):
    transform = mapping(constraints.corr_cholesky_constraint)
    x = torch.randn(x_shape, requires_grad=True).clamp(-2, 2)
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


@pytest.mark.parametrize("d", [2, 3, 4, 10])
def test_log_prob_eta1(d):
    dist = LKJCorrCholesky(d, torch.tensor([1.]))

    a_sample = dist.sample(torch.Size([100]))
    lp = dist.log_prob(a_sample)

    if d == 2:
        assert_equal(lp, lp.new_full(lp.size(), -math.log(2)))
    else:
        ladj = a_sample.diagonal(dim1=-2, dim2=-1).log().mul(
            torch.linspace(start=d-1, end=0, steps=d, device=a_sample.device, dtype=a_sample.dtype)
        ).sum(-1)
        lps_less_ladj = lp - ladj
        assert (lps_less_ladj - lps_less_ladj.min()).abs().sum() < 1e-4


@pytest.mark.parametrize("eta", [.1, .5, 1., 2., 5.])
def test_log_prob_d2(eta):
    dist = LKJCorrCholesky(2, torch.tensor([eta]))
    test_dist = TransformedDistribution(Beta(eta, eta), AffineTransform(loc=-1., scale=2.0))

    samples = dist.sample(torch.Size([100]))
    lp = dist.log_prob(samples)
    x = samples[..., 1, 0]
    tst = test_dist.log_prob(x)

    assert_tensors_equal(lp, tst, prec=1e-6)
