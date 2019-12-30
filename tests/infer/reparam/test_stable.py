import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions.torch_distribution import MaskedDistribution
from pyro.infer.reparam import StableHMMReparam, StableReparam, SymmetricStableReparam
from tests.common import assert_close


# Test helper to extract a few absolute moments from samples.
# This uses abs moments because Stable variance is infinite.
def get_moments(x):
    points = torch.tensor([-4., -1., 0., 1., 4.])
    points = points.reshape((-1,) + (1,) * x.dim())
    return torch.cat([x.mean(0, keepdim=True), (x - points).abs().mean(1)])


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
def test_stable(shape):
    stability = torch.empty(shape).uniform_(1.5, 2.).requires_grad_()
    skew = torch.empty(shape).uniform_(-0.5, 0.5).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.0).requires_grad_()
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    params = [stability, skew, scale, loc]

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 100000):
                return pyro.sample("x", dist.Stable(stability, skew, scale, loc))

    value = model()
    expected_moments = get_moments(value)

    reparam_model = poutine.reparam(model, {"x": StableReparam()})
    trace = poutine.trace(reparam_model).get_trace()
    assert isinstance(trace.nodes["x"]["fn"], MaskedDistribution)
    assert isinstance(trace.nodes["x"]["fn"].base_dist, dist.Delta)
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
        assert_close(actual_grads[3], expected_grads[3], atol=0.1)


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
def test_symmetric_stable(shape):
    stability = torch.empty(shape).uniform_(1.6, 1.9).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.0).requires_grad_()
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    params = [stability, scale, loc]

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 200000):
                return pyro.sample("x", dist.Stable(stability, 0, scale, loc))

    value = model()
    expected_moments = get_moments(value)

    reparam_model = poutine.reparam(model, {"x": SymmetricStableReparam()})
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


@pytest.mark.parametrize("duration", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("obs_dim", [1, 2])
@pytest.mark.parametrize("hidden_dim", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
def test_stable_hmm_shape(batch_shape, duration, hidden_dim, obs_dim):
    num_jumps = 3
    stability = dist.Uniform(0, 2).sample(batch_shape)

    def random_stable(shape, skew=None):
        if skew is None:
            skew = dist.Uniform(-1, 1).sample(shape)
        scale = torch.rand(shape).exp()
        loc = torch.randn(shape)
        return dist.Stable(stability, skew, scale, loc)

    init_dist = random_stable(batch_shape + (hidden_dim,), skew=0).to_event(1)
    trans_mat = torch.randn(batch_shape + (duration, hidden_dim, hidden_dim))
    trans_dist = random_stable(batch_shape + (duration, hidden_dim)).to_event(1)
    obs_mat = torch.randn(batch_shape + (duration, hidden_dim, obs_dim))
    obs_dist = random_stable(batch_shape + (duration, obs_dim), skew=0).to_event(1)
    hmm = dist.StableHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)

    def model(data=None):
        with pyro.plate_stack("plates", batch_shape):
            return pyro.sample("x", hmm, obs=data)

    data = model()
    with poutine.trace() as tr:
        with poutine.reparam(config={"x": StableHMMReparam(num_jumps)}):
            model(data)
    assert isinstance(tr.trace.nodes["x"]["fn"], dist.GaussianHMM)
    tr.trace.compute_log_prob()  # smoke test only
