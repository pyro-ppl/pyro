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


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)])
@pytest.mark.parametrize("centered", [0., 0.6, 1., torch.tensor(0.3), None])
def test_normal(centered, shape):
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.5).requires_grad_()
    if isinstance(centered, torch.Tensor):
        centered = centered.expand(shape)

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 100000):
                pyro.sample("x", dist.Normal(loc, scale))

    value = poutine.trace(model).get_trace().nodes["x"]["value"]
    expected_probe = get_moments(value)

    reparam_model = poutine.reparam(model, {"x": LocScaleReparam()})
    value = poutine.trace(reparam_model).get_trace().nodes["x"]["value"]
    actual_probe = get_moments(value)
    assert_close(actual_probe, expected_probe, atol=0.05)

    for actual_m, expected_m in zip(actual_probe, expected_probe):
        expected_grads = grad(expected_m.sum(), [loc, scale], retain_graph=True)
        actual_grads = grad(actual_m.sum(), [loc, scale], retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.02)
        assert_close(actual_grads[1], expected_grads[1], atol=0.02)
