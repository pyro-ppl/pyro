# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from scipy.stats import ks_2samp
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions.torch_distribution import MaskedDistribution
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.reparam import LatentStableReparam, StableReparam, SymmetricStableReparam
from tests.common import assert_close


# Test helper to extract a few absolute moments from univariate samples.
# This uses abs moments because Stable variance is infinite.
def get_moments(x):
    points = torch.tensor([-4., -1., 0., 1., 4.])
    points = points.reshape((-1,) + (1,) * x.dim())
    return torch.cat([x.mean(0, keepdim=True), (x - points).abs().mean(1)])


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("Reparam", [LatentStableReparam, StableReparam])
def test_stable(Reparam, shape):
    stability = torch.empty(shape).uniform_(1.5, 2.).requires_grad_()
    skew = torch.empty(shape).uniform_(-0.5, 0.5).requires_grad_()
    # test edge case when skew is 0
    if skew.dim() > 0 and skew.shape[-1] > 0:
        skew.data[..., 0] = 0.
    scale = torch.empty(shape).uniform_(0.5, 1.0).requires_grad_()
    loc = torch.empty(shape).uniform_(-1., 1.).requires_grad_()
    params = [stability, skew, scale, loc]

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 100000):
                return pyro.sample("x", dist.Stable(stability, skew, scale, loc))

    value = model()
    expected_moments = get_moments(value)

    reparam_model = poutine.reparam(model, {"x": Reparam()})
    trace = poutine.trace(reparam_model).get_trace()
    if Reparam is LatentStableReparam:
        assert isinstance(trace.nodes["x"]["fn"], MaskedDistribution)
        assert isinstance(trace.nodes["x"]["fn"].base_dist, dist.Delta)
    else:
        assert isinstance(trace.nodes["x"]["fn"], dist.Normal)
    trace.compute_log_prob()  # smoke test only
    value = trace.nodes["x"]["value"]
    actual_moments = get_moments(value)
    assert_close(actual_moments, expected_moments, atol=0.05)

    for actual_m, expected_m in zip(actual_moments, expected_moments):
        expected_grads = grad(expected_m.sum(), params, retain_graph=True)
        actual_grads = grad(actual_m.sum(), params, retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.2)
        assert_close(actual_grads[1][skew != 0], expected_grads[1][skew != 0], atol=0.1)
        assert_close(actual_grads[1][skew == 0], expected_grads[1][skew == 0], atol=0.3)
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


@pytest.mark.parametrize("skew", [-1.0, -0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("stability", [0.1, 0.4, 0.8, 0.99, 1.0, 1.01, 1.3, 1.7, 2.0])
@pytest.mark.parametrize("Reparam", [LatentStableReparam, SymmetricStableReparam, StableReparam])
def test_distribution(stability, skew, Reparam):
    if Reparam is SymmetricStableReparam and (skew != 0 or stability == 2):
        pytest.skip()
    if stability == 2 and skew in (-1, 1):
        pytest.skip()

    def model():
        with pyro.plate("particles", 20000):
            return pyro.sample("x", dist.Stable(stability, skew))

    expected = model()
    with poutine.reparam(config={"x": Reparam()}):
        actual = model()
    assert ks_2samp(expected, actual).pvalue > 0.05


@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize("Reparam", [LatentStableReparam, SymmetricStableReparam, StableReparam])
def test_subsample_smoke(Reparam, subsample):
    def model():
        with pyro.plate("plate", 10):
            with poutine.reparam(config={"x": Reparam()}):
                return pyro.sample("x", dist.Stable(1.5, 0))

    def create_plates():
        return pyro.plate("plate", 10, subsample_size=3)

    guide = AutoNormal(model, create_plates=create_plates if subsample else None)
    Trace_ELBO().loss(model, guide)  # smoke test
