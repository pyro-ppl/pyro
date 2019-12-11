import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions.reparameterize import LocScaleReparameterizer, TrivialReparameterizer
from pyro.distributions.stable import StableReparameterizer
from tests.common import assert_close


# Test helper to extract a few central moments from samples.
def normal_probe(x):
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
@pytest.mark.parametrize("Reparam", [
    TrivialReparameterizer,
    LocScaleReparameterizer,
])
def test_normal(shape, Reparam):
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.5).requires_grad_()

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 100000):
                pyro.sample("x", dist.Normal(loc, scale))

    value = poutine.trace(model).get_trace().nodes["x"]["value"]
    expected_probe = normal_probe(value)

    def config_fn(site):
        if site["name"] == "x" and isinstance(site["fn"], dist.Normal):
            return {"reparam": Reparam()}
        return {}

    reparam_model = poutine.reparam(poutine.infer_config(model, config_fn))
    value = poutine.trace(reparam_model).get_trace().nodes["x"]["value"]
    actual_probe = normal_probe(value)
    assert_close(actual_probe, expected_probe, atol=0.05)

    for actual_m, expected_m in zip(actual_probe, expected_probe):
        expected_grads = grad(expected_m.sum(), [loc, scale], retain_graph=True)
        actual_grads = grad(actual_m.sum(), [loc, scale], retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.02)
        assert_close(actual_grads[1], expected_grads[1], atol=0.02)


# Test helper to extract a few absolute moments from samples.
# This is needed for Stable because variance is infinite.
def stable_probe(x):
    points = torch.tensor([-4., -2, -1., -0.5, 0., 0.5, 1., 2, 4.])
    points = points.reshape((-1,) + (1,) * x.dim())
    return (x - points).abs().mean(1)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)])
@pytest.mark.parametrize("Reparam", [
    TrivialReparameterizer,
    StableReparameterizer,
])
def test_stable(shape, Reparam):
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
    expected_probe = stable_probe(value)

    def config_fn(site):
        if site["name"] == "x" and isinstance(site["fn"], dist.Stable):
            return {"reparam": Reparam()}
        return {}

    reparam_model = poutine.reparam(poutine.infer_config(model, config_fn))
    trace = poutine.trace(reparam_model).get_trace()
    if Reparam is StableReparameterizer:
        trace.compute_log_prob()  # smoke test only
    value = trace.nodes["x"]["value"]
    actual_probe = stable_probe(value)
    assert_close(actual_probe, expected_probe, atol=0.05)

    for actual_m, expected_m in zip(actual_probe, expected_probe):
        expected_grads = grad(expected_m.sum(), params, retain_graph=True)
        actual_grads = grad(actual_m.sum(), params, retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.2)
        assert_close(actual_grads[1], expected_grads[1], atol=0.1)
        assert_close(actual_grads[2], expected_grads[2], atol=0.1)
        assert_close(actual_grads[3], expected_grads[3], atol=0.1)
