# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.distributions import (
    AffineTransform,
    Beta,
    TransformedDistribution,
    biject_to,
    transform_to,
)

from pyro.distributions import constraints, transforms
from pyro.distributions.torch import LKJCholesky
from tests.common import assert_equal, assert_tensors_equal


@pytest.mark.parametrize("value_shape", [(1, 1), (3, 3), (5, 5)])
def test_constraint(value_shape):
    value = torch.randn(value_shape).clamp(-2, 2).tril()
    value.diagonal(dim1=-2, dim2=-1).exp_()
    value = value / value.norm(2, dim=-1, keepdim=True)

    assert (constraints.corr_cholesky.check(value) == 1).all()


def _autograd_log_det(ys, x):
    # computes log_abs_det_jacobian of y w.r.t. x
    return (
        torch.stack([torch.autograd.grad(y, (x,), retain_graph=True)[0] for y in ys])
        .det()
        .abs()
        .log()
    )


@pytest.mark.parametrize("y_shape", [(1,), (3, 1), (6,), (1, 6), (2, 6)])
def test_unconstrained_to_corr_cholesky_transform(y_shape):
    transform = transforms.CorrCholeskyTransform()
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
    transform = mapping(constraints.corr_cholesky)
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


@pytest.mark.parametrize("dim", [2, 3, 4, 10])
def test_log_prob_conc1(dim):
    dist = LKJCholesky(dim, torch.tensor([1.0]))

    a_sample = dist.sample(torch.Size([100]))
    lp = dist.log_prob(a_sample)

    if dim == 2:
        assert_equal(lp, lp.new_full(lp.size(), -math.log(2)))
    else:
        ladj = (
            a_sample.diagonal(dim1=-2, dim2=-1)
            .log()
            .mul(
                torch.linspace(
                    start=dim - 1,
                    end=0,
                    steps=dim,
                    device=a_sample.device,
                    dtype=a_sample.dtype,
                )
            )
            .sum(-1)
        )
        lps_less_ladj = lp - ladj
        assert (lps_less_ladj - lps_less_ladj.min()).abs().sum() < 1e-4


@pytest.mark.parametrize("concentration", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_log_prob_d2(concentration):
    dist = LKJCholesky(2, torch.tensor([concentration]))
    test_dist = TransformedDistribution(
        Beta(concentration, concentration), AffineTransform(loc=-1.0, scale=2.0)
    )

    samples = dist.sample(torch.Size([100]))
    lp = dist.log_prob(samples)
    x = samples[..., 1, 0]
    tst = test_dist.log_prob(x)
    # LKJ prevents inf values in log_prob
    lp[tst == math.inf] = math.inf  # substitute inf for comparison
    assert_tensors_equal(lp, tst, prec=1e-3)


def test_sample_batch():
    # Regression test for https://github.com/pyro-ppl/pyro/issues/2615
    dist = LKJCholesky(3, concentration=torch.ones(())).expand([12])
    # batch shape and event shape are as you'd expect
    assert dist.batch_shape == torch.Size([12])
    assert dist.event_shape == torch.Size([3, 3])
    # samples have correct shape when sample_shape=()
    assert dist.shape(()) == torch.Size([12, 3, 3])
    assert dist.sample().shape == torch.Size([12, 3, 3])
    # samples had the wrong shape when sample_shape is non-unit
    assert dist.shape((4,)) == torch.Size([4, 12, 3, 3])
    assert dist.sample((4,)).shape == torch.Size([4, 12, 3, 3])
