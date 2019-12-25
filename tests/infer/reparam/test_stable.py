import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.reparam import StableReparam
from tests.common import assert_close


# Test helper to extract a few absolute moments from samples.
# This uses abs moments because Stable variance is infinite.
def get_moments(x):
    points = torch.tensor([-4., -2, -1., -0.5, 0., 0.5, 1., 2, 4.])
    points = points.reshape((-1,) + (1,) * x.dim())
    return (x - points).abs().mean(1)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)])
def test_stable(shape):
    stability = torch.empty(shape).uniform_(1.5, 2.).requires_grad_()
    skew = torch.empty(shape).uniform_(-0.5, 0.5).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.0).requires_grad_()
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    params = [stability, skew, scale, loc]

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 100000):
                pyro.sample("x", dist.Stable(stability, skew, scale, loc))

    value = poutine.trace(model).get_trace().nodes["x"]["value"]
    expected_probe = get_moments(value)

    reparam_model = poutine.reparam(model, {"x": StableReparam()})
    trace = poutine.trace(reparam_model).get_trace()
    trace.compute_log_prob()  # smoke test only
    value = trace.nodes["x"]["value"]
    actual_probe = get_moments(value)
    assert_close(actual_probe, expected_probe, atol=0.05)

    for actual_m, expected_m in zip(actual_probe, expected_probe):
        expected_grads = grad(expected_m.sum(), params, retain_graph=True)
        actual_grads = grad(actual_m.sum(), params, retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.2)
        assert_close(actual_grads[1], expected_grads[1], atol=0.1)
        assert_close(actual_grads[2], expected_grads[2], atol=0.1)
        assert_close(actual_grads[3], expected_grads[3], atol=0.1)
