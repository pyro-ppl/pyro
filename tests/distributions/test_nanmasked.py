# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from tests.common import assert_close


@pytest.mark.parametrize("batch_shape", [(), (40,), (11, 9)], ids=str)
def test_normal(batch_shape):
    # Test on full data
    data = torch.randn(batch_shape)
    loc = torch.randn(batch_shape).requires_grad_()
    scale = torch.randn(batch_shape).exp().requires_grad_()
    d = dist.NanMaskedNormal(loc, scale)
    d2 = dist.Normal(loc, scale)
    actual = d.log_prob(data)
    expected = d2.log_prob(data)
    assert_close(actual, expected)

    # Test on partial data.
    ok = torch.rand(batch_shape) < 0.5
    data[~ok] = math.nan
    actual = d.log_prob(data)
    assert actual.shape == expected.shape
    assert actual.isfinite().all()
    loc_grad, scale_grad = torch.autograd.grad(actual.sum(), [loc, scale])
    assert loc_grad.isfinite().all()
    assert scale_grad.isfinite().all()

    # Check identity on fully observed and fully unobserved rows.
    assert_close(actual[ok], expected[ok])
    assert_close(actual[~ok], torch.zeros_like(actual[~ok]))


@pytest.mark.parametrize("batch_shape", [(), (40,), (11, 9)], ids=str)
@pytest.mark.parametrize("p", [1, 2, 3, 10], ids=str)
def test_multivariate_normal(batch_shape, p):
    # Test on full data
    data = torch.randn(batch_shape + (p,))
    loc = torch.randn(batch_shape + (p,)).requires_grad_()
    scale_tril = torch.randn(batch_shape + (p, p))
    scale_tril.tril_()
    scale_tril.diagonal(dim1=-2, dim2=-1).exp_()
    scale_tril.requires_grad_()
    d = dist.NanMaskedMultivariateNormal(loc, scale_tril=scale_tril)
    d2 = dist.MultivariateNormal(loc, scale_tril=scale_tril)
    actual = d.log_prob(data)
    expected = d2.log_prob(data)
    assert_close(actual, expected)

    # Test on partial data.
    ok = torch.rand(batch_shape + (p,)) < 0.5
    data[~ok] = math.nan
    actual = d.log_prob(data)
    assert actual.shape == expected.shape
    assert actual.isfinite().all()
    loc_grad, scale_tril_grad = torch.autograd.grad(actual.sum(), [loc, scale_tril])
    assert loc_grad.isfinite().all()
    assert scale_tril_grad.isfinite().all()

    # Check identity on fully observed and fully unobserved rows.
    observed = ok.all(-1)
    assert_close(actual[observed], expected[observed])
    unobserved = ~ok.any(-1)
    assert_close(actual[unobserved], torch.zeros_like(actual[unobserved]))


def test_multivariate_normal_model():
    def model(data):
        loc = pyro.sample("loc", dist.Normal(torch.zeros(3), torch.ones(3)).to_event(1))
        scale_tril = torch.eye(3)
        with pyro.plate("data", len(data)):
            pyro.sample(
                "obs",
                dist.NanMaskedMultivariateNormal(loc, scale_tril=scale_tril),
                obs=data,
            )

    data = torch.randn(100, 3)
    ok = torch.rand(100, 3) < 0.5
    assert 100 < ok.long().sum() < 200, "weak test"
    data[~ok] = math.nan

    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam({"lr": 1e-4}), Trace_ELBO())
    for step in range(3):
        loss = svi.step(data)
        assert math.isfinite(loss)
