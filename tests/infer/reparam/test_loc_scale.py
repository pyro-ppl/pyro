# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.reparam import LocScaleReparam
from tests.common import assert_close


# Test helper to extract a few central moments from samples.
def get_moments(x):
    m1 = x.mean(0)
    x = x - m1
    xx = x * x
    xxx = x * xx
    xxxx = xx * xx
    m2 = xx.mean(0)
    m3 = xxx.mean(0) / m2 ** 1.5
    m4 = xxxx.mean(0) / m2 ** 2
    return torch.stack([m1, m2, m3, m4])


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("centered", [0., 0.6, 1., torch.tensor(0.4), None])
@pytest.mark.parametrize("dist_type", ["Normal", "StudentT"])
def test_normal(dist_type, centered, shape):
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.5).requires_grad_()
    if isinstance(centered, torch.Tensor):
        centered = centered.expand(shape)

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 200000):
                if "dist_type" == "Normal":
                    pyro.sample("x", dist.Normal(loc, scale))
                else:
                    pyro.sample("x", dist.StudentT(10.0, loc, scale))

    value = poutine.trace(model).get_trace().nodes["x"]["value"]
    expected_probe = get_moments(value)

    if "dist_type" == "Normal":
        reparam = LocScaleReparam()
    else:
        reparam = LocScaleReparam(shape_params=["df"])
    reparam_model = poutine.reparam(model, {"x": reparam})
    value = poutine.trace(reparam_model).get_trace().nodes["x"]["value"]
    actual_probe = get_moments(value)
    assert_close(actual_probe, expected_probe, atol=0.1)

    for actual_m, expected_m in zip(actual_probe, expected_probe):
        expected_grads = grad(expected_m.sum(), [loc, scale], retain_graph=True)
        actual_grads = grad(actual_m.sum(), [loc, scale], retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.05)
        assert_close(actual_grads[1], expected_grads[1], atol=0.05)
