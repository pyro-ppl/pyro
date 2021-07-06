# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions.transforms.haar import HaarTransform
from pyro.infer.autoguide.initialization import InitMessenger, init_to_value
from pyro.infer.reparam import HaarReparam
from tests.common import assert_close

from .util import check_init_reparam


# Test helper to extract central moments from samples.
def get_moments(x):
    n = x.size(0)
    x = x.reshape(n, -1)
    mean = x.mean(0)
    x = x - mean
    std = (x * x).mean(0).sqrt()
    x = x / std
    corr = (x.unsqueeze(-1) * x.unsqueeze(-2)).mean(0).reshape(-1)
    return torch.cat([mean, std, corr])


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((6,), -1),
        (
            (
                2,
                5,
            ),
            -1,
        ),
        ((4, 2), -2),
        ((2, 3, 1), -2),
    ],
    ids=str,
)
def test_normal(shape, dim, flip):
    loc = torch.empty(shape).uniform_(-1.0, 1.0).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.5).requires_grad_()

    def model():
        with pyro.plate_stack("plates", shape[:dim]):
            with pyro.plate("particles", 10000):
                pyro.sample("x", dist.Normal(loc, scale).expand(shape).to_event(-dim))

    value = poutine.trace(model).get_trace().nodes["x"]["value"]
    expected_probe = get_moments(value)

    rep = HaarReparam(dim=dim, flip=flip)
    reparam_model = poutine.reparam(model, {"x": rep})
    trace = poutine.trace(reparam_model).get_trace()
    assert isinstance(trace.nodes["x_haar"]["fn"], dist.TransformedDistribution)
    assert isinstance(trace.nodes["x"]["fn"], dist.Delta)
    value = trace.nodes["x"]["value"]
    actual_probe = get_moments(value)
    assert_close(actual_probe, expected_probe, atol=0.1)

    for actual_m, expected_m in zip(actual_probe[:10], expected_probe[:10]):
        expected_grads = grad(expected_m.sum(), [loc, scale], retain_graph=True)
        actual_grads = grad(actual_m.sum(), [loc, scale], retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.05)
        assert_close(actual_grads[1], expected_grads[1], atol=0.05)


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((6,), -1),
        (
            (
                2,
                5,
            ),
            -1,
        ),
        ((4, 2), -2),
        ((2, 3, 1), -2),
    ],
    ids=str,
)
def test_uniform(shape, dim, flip):
    def model():
        with pyro.plate_stack("plates", shape[:dim]):
            with pyro.plate("particles", 10000):
                pyro.sample("x", dist.Uniform(0, 1).expand(shape).to_event(-dim))

    value = poutine.trace(model).get_trace().nodes["x"]["value"]
    expected_probe = get_moments(value)

    reparam_model = poutine.reparam(model, {"x": HaarReparam(dim=dim, flip=flip)})
    trace = poutine.trace(reparam_model).get_trace()
    assert isinstance(trace.nodes["x_haar"]["fn"], dist.TransformedDistribution)
    assert isinstance(trace.nodes["x"]["fn"], dist.Delta)
    value = trace.nodes["x"]["value"]
    actual_probe = get_moments(value)
    assert_close(actual_probe, expected_probe, atol=0.1)


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((6,), -1),
        (
            (
                2,
                5,
            ),
            -1,
        ),
        ((4, 2), -2),
        ((2, 3, 1), -2),
    ],
    ids=str,
)
def test_init(shape, dim, flip):
    loc = torch.empty(shape).uniform_(-1.0, 1.0).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.5).requires_grad_()

    def model():
        with pyro.plate_stack("plates", shape[:dim]):
            return pyro.sample("x", dist.Normal(loc, scale).to_event(-dim))

    check_init_reparam(model, HaarReparam(dim=dim, flip=flip))


def test_nested():
    shape = (5, 6)

    @poutine.reparam(config={"x": HaarReparam(dim=-1), "x_haar": HaarReparam(dim=-2)})
    def model():
        pyro.sample("x", dist.Normal(torch.zeros(shape), 1).to_event(2))

    # Try without initialization, e.g. in AutoGuide._setup_prototype().
    trace = poutine.trace(model).get_trace()
    assert {"x", "x_haar", "x_haar_haar"}.issubset(trace.nodes)
    assert trace.nodes["x"]["is_observed"]
    assert trace.nodes["x_haar"]["is_observed"]
    assert not trace.nodes["x_haar_haar"]["is_observed"]
    assert trace.nodes["x"]["value"].shape == shape

    # Try conditioning on x_haar_haar, e.g. in Predictive.
    x = torch.randn(shape)
    x_haar = HaarTransform(dim=-1)(x)
    x_haar_haar = HaarTransform(dim=-2)(x_haar)
    with poutine.condition(data={"x_haar_haar": x_haar_haar}):
        trace = poutine.trace(model).get_trace()
        assert {"x", "x_haar", "x_haar_haar"}.issubset(trace.nodes)
        assert trace.nodes["x"]["is_observed"]
        assert trace.nodes["x_haar"]["is_observed"]
        assert trace.nodes["x_haar_haar"]["is_observed"]
        assert_close(trace.nodes["x"]["value"], x)
        assert_close(trace.nodes["x_haar"]["value"], x_haar)
        assert_close(trace.nodes["x_haar_haar"]["value"], x_haar_haar)

    # Try with custom initialization.
    # This is required for autoguides and MCMC.
    with InitMessenger(init_to_value(values={"x": x})):
        trace = poutine.trace(model).get_trace()
        assert {"x", "x_haar", "x_haar_haar"}.issubset(trace.nodes)
        assert trace.nodes["x"]["is_observed"]
        assert trace.nodes["x_haar"]["is_observed"]
        assert not trace.nodes["x_haar_haar"]["is_observed"]
        assert_close(trace.nodes["x"]["value"], x)

    # Try conditioning on x.
    x = torch.randn(shape)
    with poutine.condition(data={"x": x}):
        trace = poutine.trace(model).get_trace()
        assert {"x", "x_haar", "x_haar_haar"}.issubset(trace.nodes)
        assert trace.nodes["x"]["is_observed"]
        assert trace.nodes["x_haar"]["is_observed"]
        # TODO Decide whether it is worth fixing this failing assertion.
        # See https://github.com/pyro-ppl/pyro/issues/2878
        # assert trace.nodes["x_haar_haar"]["is_observed"]
        assert_close(trace.nodes["x"]["value"], x)
