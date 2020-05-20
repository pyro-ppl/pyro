# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions.transforms import Permute
from pyro.infer.reparam import UnitJacobianReparam
from tests.common import assert_close


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


@pytest.mark.parametrize("shape", [(6,), (4, 5), (2, 1, 3)], ids=str)
def test_normal(shape):
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.5).requires_grad_()

    def model():
        with pyro.plate_stack("plates", shape[:-1]):
            with pyro.plate("particles", 10000):
                pyro.sample("x", dist.Normal(loc, scale).expand(shape).to_event(1))

    value = poutine.trace(model).get_trace().nodes["x"]["value"]
    expected_probe = get_moments(value)

    transform = Permute(torch.randperm(shape[-1]))
    rep = UnitJacobianReparam(transform)
    reparam_model = poutine.reparam(model, {"x": rep})
    trace = poutine.trace(reparam_model).get_trace()
    assert isinstance(trace.nodes["x_transformed"]["fn"], dist.TransformedDistribution)
    assert isinstance(trace.nodes["x"]["fn"], dist.Delta)
    value = trace.nodes["x"]["value"]
    actual_probe = get_moments(value)
    assert_close(actual_probe, expected_probe, atol=0.1)

    for actual_m, expected_m in zip(actual_probe[:10], expected_probe[:10]):
        expected_grads = grad(expected_m.sum(), [loc, scale], retain_graph=True)
        actual_grads = grad(actual_m.sum(), [loc, scale], retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.05)
        assert_close(actual_grads[1], expected_grads[1], atol=0.05)
