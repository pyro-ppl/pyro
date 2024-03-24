# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch
from scipy.stats import ks_2samp
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions import constraints
from pyro.distributions.torch_distribution import MaskedDistribution
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.mcmc import MCMC, NUTS
from pyro.infer.reparam import (
    LatentStableReparam,
    StableReparam,
    SymmetricStableReparam,
)
from pyro.optim import ClippedAdam
from tests.common import assert_close

from .util import check_init_reparam

logger = logging.getLogger(__name__)


# Test helper to extract a few absolute moments from univariate samples.
# This uses abs moments because Stable variance is infinite.
def get_moments(x):
    points = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
    points = points.reshape((-1,) + (1,) * x.dim())
    return torch.cat([x.mean(0, keepdim=True), (x - points).abs().mean(1)])


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("Reparam", [LatentStableReparam, StableReparam])
def test_stable(Reparam, shape):
    stability = torch.empty(shape).uniform_(1.5, 2.0).requires_grad_()
    skew = torch.empty(shape).uniform_(-0.5, 0.5).requires_grad_()
    # test edge case when skew is 0
    if skew.dim() > 0 and skew.shape[-1] > 0:
        skew.data[..., 0] = 0.0
    scale = torch.empty(shape).uniform_(0.5, 1.0).requires_grad_()
    loc = torch.empty(shape).uniform_(-1.0, 1.0).requires_grad_()
    params = [stability, skew, scale, loc]

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 200000):
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
    assert_close(actual_moments, expected_moments, atol=0.07)

    for actual_m, expected_m in zip(actual_moments, expected_moments):
        expected_grads = grad(expected_m.sum(), params, retain_graph=True)
        actual_grads = grad(actual_m.sum(), params, retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.3)
        assert_close(actual_grads[1][skew != 0], expected_grads[1][skew != 0], atol=0.1)
        assert_close(actual_grads[1][skew == 0], expected_grads[1][skew == 0], atol=0.3)
        assert_close(actual_grads[2], expected_grads[2], atol=0.1)
        assert_close(actual_grads[3], expected_grads[3], atol=0.1)


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)], ids=str)
def test_symmetric_stable(shape):
    stability = torch.empty(shape).uniform_(1.6, 1.9).requires_grad_()
    scale = torch.empty(shape).uniform_(0.5, 1.0).requires_grad_()
    loc = torch.empty(shape).uniform_(-1.0, 1.0).requires_grad_()
    params = [stability, scale, loc]

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 300000):
                return pyro.sample("x", dist.Stable(stability, 0, scale, loc))

    value = model()
    expected_moments = get_moments(value)

    reparam_model = poutine.reparam(model, {"x": SymmetricStableReparam()})
    trace = poutine.trace(reparam_model).get_trace()
    assert isinstance(trace.nodes["x"]["fn"], dist.Normal)
    trace.compute_log_prob()  # smoke test only
    value = trace.nodes["x"]["value"]
    actual_moments = get_moments(value)
    assert_close(actual_moments, expected_moments, atol=0.07)

    for actual_m, expected_m in zip(actual_moments, expected_moments):
        expected_grads = grad(expected_m.sum(), params, retain_graph=True)
        actual_grads = grad(actual_m.sum(), params, retain_graph=True)
        assert_close(actual_grads[0], expected_grads[0], atol=0.2)
        assert_close(actual_grads[1], expected_grads[1], atol=0.1)
        assert_close(actual_grads[2], expected_grads[2], atol=0.1)


@pytest.mark.parametrize("skew", [-1.0, -0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("stability", [0.1, 0.4, 0.8, 0.99, 1.0, 1.01, 1.3, 1.7, 2.0])
@pytest.mark.parametrize(
    "Reparam", [LatentStableReparam, SymmetricStableReparam, StableReparam]
)
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
    assert ks_2samp(expected, actual).pvalue > 0.03


@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize(
    "Reparam", [LatentStableReparam, SymmetricStableReparam, StableReparam]
)
def test_subsample_smoke(Reparam, subsample):
    def model():
        with poutine.reparam(config={"x": Reparam()}):
            with pyro.plate("plate", 10):
                return pyro.sample("x", dist.Stable(1.5, 0))

    def create_plates():
        return pyro.plate("plate", 10, subsample_size=3)

    guide = AutoNormal(model, create_plates=create_plates if subsample else None)
    Trace_ELBO().loss(model, guide)  # smoke test


@pytest.mark.parametrize("skew", [-1.0, -0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("stability", [0.1, 0.4, 0.8, 0.99, 1.0, 1.01, 1.3, 1.7, 2.0])
@pytest.mark.parametrize(
    "Reparam",
    [LatentStableReparam, SymmetricStableReparam, StableReparam],
)
def test_init(stability, skew, Reparam):
    if Reparam is SymmetricStableReparam and (skew != 0 or stability == 2):
        pytest.skip()
    if stability == 2 and skew in (-1, 1):
        pytest.skip()

    def model():
        return pyro.sample("x", dist.Stable(stability, skew))

    check_init_reparam(model, Reparam())


@pytest.mark.stage("integration", "integration_batch_1")
@pytest.mark.parametrize(
    "stability, skew, scale, loc",
    [
        (1.9, 0.0, 2.0, 1.0),
        (0.8, 0.0, 3.0, 2.0),
    ],
)
def test_symmetric_stable_mle(stability, skew, scale, loc):
    # Regression test for https://github.com/pyro-ppl/pyro/issues/3280
    assert skew == 0.0
    data = dist.Stable(stability, skew, scale, loc).sample([10000])

    @poutine.reparam(config={"x": SymmetricStableReparam()})
    def mle_model():
        a = pyro.param("a", torch.tensor(1.9), constraint=constraints.interval(0, 2))
        b = 0.0
        c = pyro.param("c", torch.tensor(1.0), constraint=constraints.positive)
        d = pyro.param("d", torch.tensor(0.0), constraint=constraints.real)
        with pyro.plate("data", len(data)):
            pyro.sample("x", dist.Stable(a, b, c, d), obs=data)

    num_steps = 1001
    guide = AutoNormal(mle_model)
    optim = ClippedAdam({"clip_norm": 100, "lr": 0.05, "lrd": 0.1 ** (1 / num_steps)})
    svi = SVI(mle_model, guide, optim, Trace_ELBO())
    for step in range(num_steps):
        loss = svi.step() / len(data)
        if step % 100 == 0:
            logger.info("step %d loss = %g", step, loss)

    # Check loss against a true model.
    @poutine.reparam(config={"x": SymmetricStableReparam()})
    def true_model():
        with pyro.plate("data", len(data)):
            pyro.sample("x", dist.Stable(stability, skew, scale, loc), obs=data)

    actual_loss = Trace_ELBO().loss(mle_model, guide) / len(data)
    expected_loss = Trace_ELBO().loss(true_model, guide) / len(data)
    assert_close(actual_loss, expected_loss, atol=0.33)

    # Check parameter estimates.
    actual = {name: float(pyro.param(name).data) for name in "acd"}
    assert_close(actual["a"], stability, atol=0.1)
    assert_close(actual["c"], scale, atol=0.1, rtol=0.1)
    assert_close(actual["d"], loc, atol=0.1)


@pytest.mark.stage("integration", "integration_batch_1")
@pytest.mark.parametrize(
    "stability, skew, scale, loc",
    [
        (1.9, 0.0, 2.0, 1.0),
        (0.8, 0.0, 3.0, 2.0),
        (1.8, 0.8, 4.0, 3.0),
    ],
)
def test_stable_mle(stability, skew, scale, loc):
    # Regression test for https://github.com/pyro-ppl/pyro/issues/3280
    data = dist.Stable(stability, skew, scale, loc).sample([10000])

    @poutine.reparam(config={"x": StableReparam()})
    def mle_model():
        a = pyro.param("a", torch.tensor(1.9), constraint=constraints.interval(0, 2))
        b = pyro.param("b", torch.tensor(0.0), constraint=constraints.interval(-1, 1))
        c = pyro.param("c", torch.tensor(1.0), constraint=constraints.positive)
        d = pyro.param("d", torch.tensor(0.0), constraint=constraints.real)
        with pyro.plate("data", len(data)):
            pyro.sample("x", dist.Stable(a, b, c, d), obs=data)

    num_steps = 1001
    guide = AutoNormal(mle_model)
    optim = ClippedAdam({"clip_norm": 100, "lr": 0.05, "lrd": 0.1 ** (1 / num_steps)})
    svi = SVI(mle_model, guide, optim, Trace_ELBO())
    for step in range(num_steps):
        loss = svi.step() / len(data)
        if step % 100 == 0:
            logger.info("step %d loss = %g", step, loss)

    # Check loss against a true model.
    @poutine.reparam(config={"x": StableReparam()})
    def true_model():
        with pyro.plate("data", len(data)):
            pyro.sample("x", dist.Stable(stability, skew, scale, loc), obs=data)

    actual_loss = Trace_ELBO().loss(mle_model, guide) / len(data)
    expected_loss = Trace_ELBO().loss(true_model, guide) / len(data)
    assert_close(actual_loss, expected_loss, atol=0.1)

    # Check parameter estimates.
    actual = {name: float(pyro.param(name).data) for name in "abcd"}
    assert_close(actual["a"], stability, atol=0.1)
    assert_close(actual["b"], skew, atol=0.1)
    assert_close(actual["c"], scale, atol=0.1, rtol=0.1)
    assert_close(actual["d"], loc, atol=0.1)


@pytest.mark.stage("integration", "integration_batch_1")
@pytest.mark.parametrize(
    "stability, skew, scale, loc",
    [
        (1.9, 0.0, 2.0, 1.0),
        (0.8, 0.0, 3.0, 2.0),
        (1.8, 0.8, 4.0, 3.0),
    ],
)
def test_stable_mcmc(stability, skew, scale, loc):
    # Regression test for https://github.com/pyro-ppl/pyro/issues/3280
    data = dist.Stable(stability, skew, scale, loc).sample([1000])

    @poutine.reparam(config={"x": StableReparam()})
    def model():
        with poutine.mask(mask=False):  # flat prior
            a = pyro.sample("a", dist.Uniform(0, 2))
            b = pyro.sample("b", dist.Uniform(-1, 1))
            c = pyro.sample("c", dist.Exponential(1))
            d = pyro.sample("d", dist.Normal(0, 1))
        with pyro.plate("data", len(data)):
            pyro.sample("x", dist.Stable(a, b, c, d), obs=data)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=400, warmup_steps=200)
    mcmc.run()
    samples = mcmc.get_samples()
    actual = {k: v.mean().item() for k, v in samples.items()}
    assert_close(actual["a"], stability, atol=0.1)
    assert_close(actual["b"], skew, atol=0.1)
    assert_close(actual["c"], scale, atol=0.1, rtol=0.1)
    assert_close(actual["d"], loc, atol=0.1)
