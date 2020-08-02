# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.distributions import (Delta, NegativeBinomial, Normal, Poisson, ZeroInflatedDistribution,
                                ZeroInflatedNegativeBinomial, ZeroInflatedPoisson)
from pyro.distributions.util import broadcast_shape
from tests.common import assert_close


@pytest.mark.parametrize("gate_shape", [(), (2,), (3, 1), (3, 2)])
@pytest.mark.parametrize("base_shape", [(), (2,), (3, 1), (3, 2)])
def test_zid_shape(gate_shape, base_shape):
    gate = torch.rand(gate_shape)
    base_dist = Normal(torch.randn(base_shape), torch.randn(base_shape).exp())

    d = ZeroInflatedDistribution(gate, base_dist)
    assert d.batch_shape == broadcast_shape(gate_shape, base_shape)
    assert d.support == base_dist.support

    d2 = d.expand([4, 3, 2])
    assert d2.batch_shape == (4, 3, 2)


@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_0_gate(rate):
    # if gate is 0 ZIP is Poisson
    zip_ = ZeroInflatedPoisson(torch.zeros(1), torch.tensor(rate))
    pois = Poisson(torch.tensor(rate))
    s = pois.sample((20,))
    zip_prob = zip_.log_prob(s)
    pois_prob = pois.log_prob(s)
    assert_close(zip_prob, pois_prob)


@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_1_gate(rate):
    # if gate is 1 ZIP is Delta(0)
    zip_ = ZeroInflatedPoisson(torch.ones(1), torch.tensor(rate))
    delta = Delta(torch.zeros(1))
    s = torch.tensor([0.0, 1.0])
    zip_prob = zip_.log_prob(s)
    delta_prob = delta.log_prob(s)
    assert_close(zip_prob, delta_prob)


@pytest.mark.parametrize("gate", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_zip_mean_variance(gate, rate):
    num_samples = 1000000
    zip_ = ZeroInflatedPoisson(torch.tensor(gate), torch.tensor(rate))
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
    zinb_ = ZeroInflatedNegativeBinomial(
        torch.zeros(1), total_count=torch.tensor(total_count), probs=torch.tensor(probs)
    )
    neg_bin = NegativeBinomial(torch.tensor(total_count), probs=torch.tensor(probs))
    s = neg_bin.sample((20,))
    zinb_prob = zinb_.log_prob(s)
    neg_bin_prob = neg_bin.log_prob(s)
    assert_close(zinb_prob, neg_bin_prob)


@pytest.mark.parametrize("total_count", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
@pytest.mark.parametrize("probs", [0.1, 0.5, 0.9])
def test_zinb_1_gate(total_count, probs):
    # if gate is 1 ZINB is Delta(0)
    zinb_ = ZeroInflatedNegativeBinomial(
        torch.ones(1), total_count=torch.tensor(total_count), probs=torch.tensor(probs)
    )
    delta = Delta(torch.zeros(1))
    s = torch.tensor([0.0, 1.0])
    zinb_prob = zinb_.log_prob(s)
    delta_prob = delta.log_prob(s)
    assert_close(zinb_prob, delta_prob)


@pytest.mark.parametrize("gate", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("total_count", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
@pytest.mark.parametrize("logits", [-0.5, 0.5, -0.9, 1.9])
def test_zinb_mean_variance(gate, total_count, logits):
    num_samples = 1000000
    zinb_ = ZeroInflatedNegativeBinomial(
        torch.tensor(gate),
        total_count=torch.tensor(total_count),
        logits=torch.tensor(logits),
    )
    s = zinb_.sample((num_samples,))
    expected_mean = zinb_.mean
    estimated_mean = s.mean()
    expected_std = zinb_.stddev
    estimated_std = s.std()
    assert_close(expected_mean, estimated_mean, atol=1e-01)
    assert_close(expected_std, estimated_std, atol=1e-1)
