# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import pyro
from pyro.distributions import (
    Delta,
    NegativeBinomial,
    Normal,
    Poisson,
    ZeroInflatedDistribution,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from pyro.distributions.util import broadcast_shape
from tests.common import assert_close


@pytest.mark.parametrize("gate_shape", [(), (2,), (3, 1), (3, 2)])
@pytest.mark.parametrize("base_shape", [(), (2,), (3, 1), (3, 2)])
def test_zid_shape(gate_shape, base_shape):
    gate = torch.rand(gate_shape)
    base_dist = Normal(torch.randn(base_shape), torch.randn(base_shape).exp())

    d = ZeroInflatedDistribution(base_dist, gate=gate)
    assert d.batch_shape == broadcast_shape(gate_shape, base_shape)
    assert d.support == base_dist.support

    d2 = d.expand([4, 3, 2])
    assert d2.batch_shape == (4, 3, 2)


@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_0_gate(rate):
    # if gate is 0 ZIP is Poisson
    zip1 = ZeroInflatedPoisson(torch.tensor(rate), gate=torch.zeros(1))
    zip2 = ZeroInflatedPoisson(torch.tensor(rate), gate_logits=torch.tensor(-99.9))
    pois = Poisson(torch.tensor(rate))
    s = pois.sample((20,))
    zip1_prob = zip1.log_prob(s)
    zip2_prob = zip2.log_prob(s)
    pois_prob = pois.log_prob(s)
    assert_close(zip1_prob, pois_prob)
    assert_close(zip2_prob, pois_prob)


@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_1_gate(rate):
    # if gate is 1 ZIP is Delta(0)
    zip1 = ZeroInflatedPoisson(torch.tensor(rate), gate=torch.ones(1))
    zip2 = ZeroInflatedPoisson(torch.tensor(rate), gate_logits=torch.tensor(math.inf))
    delta = Delta(torch.zeros(1))
    s = torch.tensor([0.0, 1.0])
    zip1_prob = zip1.log_prob(s)
    zip2_prob = zip2.log_prob(s)
    delta_prob = delta.log_prob(s)
    assert_close(zip1_prob, delta_prob)
    assert_close(zip2_prob, delta_prob)


@pytest.mark.parametrize("gate", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_mean_variance(gate, rate):
    num_samples = 1000000
    zip_ = ZeroInflatedPoisson(torch.tensor(rate), gate=torch.tensor(gate))
    s = zip_.sample((num_samples,))
    expected_mean = zip_.mean
    estimated_mean = s.mean()
    expected_std = zip_.stddev
    estimated_std = s.std()
    assert_close(expected_mean, estimated_mean, atol=1e-02)
    assert_close(expected_std, estimated_std, atol=1e-02)


@pytest.mark.parametrize("total_count", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
@pytest.mark.parametrize("probs", [0.1, 0.5, 0.9])
def test_zinb_0_gate(total_count, probs):
    # if gate is 0 ZINB is NegativeBinomial
    zinb1 = ZeroInflatedNegativeBinomial(
        total_count=torch.tensor(total_count),
        gate=torch.zeros(1),
        probs=torch.tensor(probs),
    )
    zinb2 = ZeroInflatedNegativeBinomial(
        total_count=torch.tensor(total_count),
        gate_logits=torch.tensor(-99.9),
        probs=torch.tensor(probs),
    )
    neg_bin = NegativeBinomial(torch.tensor(total_count), probs=torch.tensor(probs))
    s = neg_bin.sample((20,))
    zinb1_prob = zinb1.log_prob(s)
    zinb2_prob = zinb2.log_prob(s)
    neg_bin_prob = neg_bin.log_prob(s)
    assert_close(zinb1_prob, neg_bin_prob)
    assert_close(zinb2_prob, neg_bin_prob)


@pytest.mark.parametrize("total_count", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
@pytest.mark.parametrize("probs", [0.1, 0.5, 0.9])
def test_zinb_1_gate(total_count, probs):
    # if gate is 1 ZINB is Delta(0)
    zinb1 = ZeroInflatedNegativeBinomial(
        total_count=torch.tensor(total_count),
        gate=torch.ones(1),
        probs=torch.tensor(probs),
    )
    zinb2 = ZeroInflatedNegativeBinomial(
        total_count=torch.tensor(total_count),
        gate_logits=torch.tensor(math.inf),
        probs=torch.tensor(probs),
    )
    delta = Delta(torch.zeros(1))
    s = torch.tensor([0.0, 1.0])
    zinb1_prob = zinb1.log_prob(s)
    zinb2_prob = zinb2.log_prob(s)
    delta_prob = delta.log_prob(s)
    assert_close(zinb1_prob, delta_prob)
    assert_close(zinb2_prob, delta_prob)


@pytest.mark.parametrize("gate", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("total_count", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
@pytest.mark.parametrize("logits", [-0.5, 0.5, -0.9, 1.9])
def test_zinb_mean_variance(gate, total_count, logits):
    num_samples = 1000000
    zinb_ = ZeroInflatedNegativeBinomial(
        total_count=torch.tensor(total_count),
        gate=torch.tensor(gate),
        logits=torch.tensor(logits),
    )
    s = zinb_.sample((num_samples,))
    expected_mean = zinb_.mean
    estimated_mean = s.mean()
    expected_std = zinb_.stddev
    estimated_std = s.std()
    assert_close(expected_mean, estimated_mean, atol=1e-01)
    assert_close(expected_std, estimated_std, atol=1e-1)


def _make_zero_inflated(kind, rate, gate_parameter, parameter, validate_args=True):
    kwargs = {parameter: gate_parameter, "validate_args": validate_args}
    if kind == "generic":
        return ZeroInflatedDistribution(Poisson(rate), **kwargs)
    if kind == "poisson":
        return ZeroInflatedPoisson(rate, **kwargs)
    return ZeroInflatedNegativeBinomial(
        rate, probs=torch.full_like(rate, 0.4), **kwargs
    )


@pytest.mark.parametrize("kind", ["generic", "poisson", "negative_binomial"])
@pytest.mark.parametrize("parameter", ["gate", "gate_logits"])
@pytest.mark.parametrize(
    "accesses",
    [
        (),
        ("mean",),
        ("variance",),
        ("sample",),
        ("log_prob",),
        ("gate",),
        ("gate_logits",),
        ("gate_logits", "gate"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("validate_args", [False, True])
def test_expand_after_lazy_property(kind, parameter, accesses, dtype, validate_args):
    rate = torch.tensor([1.5, 2.0], dtype=dtype)
    gate_parameter = torch.tensor([0.2, 0.7], dtype=dtype)
    distribution = _make_zero_inflated(
        kind, rate, gate_parameter, parameter, validate_args
    )
    value = torch.tensor([0.0, 2.0], dtype=dtype)
    for access in accesses:
        if access == "sample":
            distribution.sample()
        elif access == "log_prob":
            distribution.log_prob(value)
        else:
            getattr(distribution, access)

    expanded = distribution.expand((3, 2))
    assert type(expanded) is type(distribution)
    assert expanded.batch_shape == (3, 2)
    assert expanded.event_shape == distribution.event_shape
    assert expanded._validate_args is validate_args
    for name in ("gate", "gate_logits"):
        assert (name in expanded.__dict__) == (name in distribution.__dict__)
        if name in distribution.__dict__:
            assert_close(
                expanded.__dict__[name], distribution.__dict__[name].expand(3, 2)
            )
    assert_close(expanded.log_prob(value), distribution.log_prob(value).expand(3, 2))
    assert_close(expanded.mean, distribution.mean.expand(3, 2))
    assert_close(expanded.variance, distribution.variance.expand(3, 2))
    pyro.set_rng_seed(123)
    expected = distribution.sample((3,))
    pyro.set_rng_seed(123)
    assert_close(expanded.sample(), expected)
    assert expanded.expand((4, 3, 2)).batch_shape == (4, 3, 2)


@pytest.mark.parametrize("kind", ["generic", "poisson", "negative_binomial"])
@pytest.mark.parametrize("parameter", ["gate", "gate_logits"])
@pytest.mark.parametrize("accesses", [(), ("mean",), ("gate", "gate_logits")])
def test_expand_preserves_mixture_values_and_gradients(kind, parameter, accesses):
    rate = torch.tensor([1.5, 2.0], requires_grad=True)
    gate_parameter = torch.tensor([0.2, 0.7], requires_grad=True)
    distribution = _make_zero_inflated(kind, rate, gate_parameter, parameter)
    for name in accesses:
        getattr(distribution, name)
    expanded = distribution.expand((3, 2))
    value = torch.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]])
    original_log_prob = distribution.log_prob(value)
    expanded_log_prob = expanded.log_prob(value)
    gate = gate_parameter if parameter == "gate" else gate_parameter.sigmoid()
    base = (
        torch.distributions.NegativeBinomial(rate, probs=torch.full_like(rate, 0.4))
        if kind == "negative_binomial"
        else torch.distributions.Poisson(rate)
    )
    # Dense two-component mixture: a point mass at zero and the base PMF.
    probabilities = torch.stack(
        [(value == 0).to(value.dtype), base.log_prob(value).exp()], -1
    )
    weights = torch.stack([gate, 1 - gate], -1)
    expected = (probabilities * weights).sum(-1).log()
    assert_close(expanded_log_prob, original_log_prob)
    assert_close(expanded_log_prob, expected)
    gradients = [
        torch.autograd.grad(result.sum(), (rate, gate_parameter), retain_graph=True)
        for result in (original_log_prob, expanded_log_prob, expected)
    ]
    for actual in gradients[1:]:
        for actual_gradient, expected_gradient in zip(actual, gradients[0]):
            assert_close(actual_gradient, expected_gradient)


def test_plate_after_zero_inflated_mean():
    distribution = ZeroInflatedPoisson(
        torch.tensor(2.0), gate_logits=torch.tensor(-0.5)
    )
    distribution.mean

    def model():
        with pyro.plate("data", 3):
            pyro.sample("observed", distribution, obs=torch.zeros(3))

    trace = pyro.poutine.trace(model).get_trace()
    trace.compute_log_prob()
    expected = torch.log(
        torch.sigmoid(torch.tensor(-0.5))
        + torch.sigmoid(torch.tensor(0.5)) * torch.exp(torch.tensor(-2.0))
    )
    assert_close(trace.nodes["observed"]["log_prob"], expected.expand(3))
