# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.reparam import ProjectedNormalReparam
from tests.common import assert_close

from .util import check_init_reparam


# Test helper to extract a few central moments from samples.
def get_moments(x):
    m1 = x.mean(0)
    x = x - m1
    xx = x[..., None] * x[..., None, :]
    m2 = xx.mean(0)
    return torch.cat([m1.reshape(-1), m2.reshape(-1)])


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_projected_normal(shape, dim):
    concentration = torch.randn(shape + (dim,)).requires_grad_()

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 10000):
                pyro.sample("x", dist.ProjectedNormal(concentration))

    value = poutine.trace(model).get_trace().nodes["x"]["value"]
    assert dist.ProjectedNormal.support.check(value).all()
    expected_probe = get_moments(value)

    reparam_model = poutine.reparam(model, {"x": ProjectedNormalReparam()})
    value = poutine.trace(reparam_model).get_trace().nodes["x"]["value"]
    assert dist.ProjectedNormal.support.check(value).all()
    actual_probe = get_moments(value)
    assert_close(actual_probe, expected_probe, atol=0.05)

    for actual_m, expected_m in zip(actual_probe, expected_probe):
        expected_grad = grad(expected_m, [concentration], retain_graph=True)
        actual_grad = grad(actual_m, [concentration], retain_graph=True)
        assert_close(actual_grad, expected_grad, atol=0.1)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_init(shape, dim):
    concentration = torch.randn(shape + (dim,)).requires_grad_()

    def model():
        with pyro.plate_stack("plates", shape):
            return pyro.sample("x", dist.ProjectedNormal(concentration))

    check_init_reparam(model, ProjectedNormalReparam())
