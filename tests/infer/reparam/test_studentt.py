# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from scipy.stats import ks_2samp
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.reparam import StudentTReparam
from tests.common import assert_close


# Test helper to extract a few absolute moments from univariate samples.
# This uses abs moments because StudentT variance may be infinite.
def get_moments(x):
    points = torch.tensor([-4., -1., 0., 1., 4.])
    points = points.reshape((-1,) + (1,) * x.dim())
    return torch.cat([x.mean(0, keepdim=True), (x - points).abs().mean(1)])


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
def test_moments(shape):
    df = torch.empty(shape).uniform_(1.8, 5).requires_grad_()
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.0).requires_grad_()
    params = [df, loc, scale]

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 100000):
                return pyro.sample("x", dist.StudentT(df, loc, scale))

    value = model()
    expected_moments = get_moments(value)

    reparam_model = poutine.reparam(model, {"x": StudentTReparam()})
    trace = poutine.trace(reparam_model).get_trace()
    assert isinstance(trace.nodes["x"]["fn"], dist.Normal)
    trace.compute_log_prob()  # smoke test only
    value = trace.nodes["x"]["value"]
    actual_moments = get_moments(value)
    assert_close(actual_moments, expected_moments, atol=0.05)

    for actual_m, expected_m in zip(actual_moments, expected_moments):
        expected_grads = grad(expected_m.sum(), params, retain_graph=True)
        actual_grads = grad(actual_m.sum(), params, retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.2)
        assert_close(actual_grads[1], expected_grads[1], atol=0.1)
        assert_close(actual_grads[2], expected_grads[2], atol=0.1)


@pytest.mark.parametrize("df", [0.5, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("scale", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("loc", [0.0, 1.234])
def test_distribution(df, loc, scale):

    def model():
        with pyro.plate("particles", 20000):
            return pyro.sample("x", dist.StudentT(df, loc, scale))

    expected = model()
    with poutine.reparam(config={"x": StudentTReparam()}):
        actual = model()
    assert ks_2samp(expected, actual).pvalue > 0.05
