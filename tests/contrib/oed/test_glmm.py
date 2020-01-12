# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.distributions.transforms import AffineTransform, SigmoidTransform

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.oed.glmm import (
    known_covariance_linear_model, group_linear_model, zero_mean_unit_obs_sd_lm,
    normal_inverse_gamma_linear_model, logistic_regression_model, sigmoid_model
)
from tests.common import assert_equal


def lm_2p_10_10_1(design):
    w = pyro.sample("w", dist.Normal(torch.tensor(0.),
                                     torch.tensor([10., 10.])).to_event(1))
    mean = torch.matmul(design, w.unsqueeze(-1)).squeeze(-1)
    y = pyro.sample("y", dist.Normal(mean, torch.tensor(1.)).to_event(1))
    return y


def lm_2p_10_10_1_w12(design):
    w1 = pyro.sample("w1", dist.Normal(torch.tensor([0.]),
                                       torch.tensor(10.)).to_event(1))
    w2 = pyro.sample("w2", dist.Normal(torch.tensor([0.]),
                                       torch.tensor(10.)).to_event(1))
    w = torch.cat([w1, w2], dim=-1)
    mean = torch.matmul(design, w.unsqueeze(-1)).squeeze(-1)
    y = pyro.sample("y", dist.Normal(mean, torch.tensor(1.)).to_event(1))
    return y


def nz_lm_2p_10_10_1(design):
    w = pyro.sample("w", dist.Normal(torch.tensor([1., -1.]),
                                     torch.tensor([10., 10.])).to_event(1))
    mean = torch.matmul(design, w.unsqueeze(-1)).squeeze(-1)
    y = pyro.sample("y", dist.Normal(mean, torch.tensor(1.)).to_event(1))
    return y


def normal_inv_gamma_2_2_10_10(design):
    tau = pyro.sample("tau", dist.Gamma(torch.tensor(2.), torch.tensor(2.)))
    obs_sd = 1./torch.sqrt(tau)
    w = pyro.sample("w", dist.Normal(torch.tensor([1., -1.]),
                                     obs_sd*torch.tensor([10., 10.])).to_event(1))
    mean = torch.matmul(design, w.unsqueeze(-1)).squeeze(-1)
    y = pyro.sample("y", dist.Normal(mean, torch.tensor(1.)).to_event(1))
    return y


def lr_10_10(design):
    w = pyro.sample("w", dist.Normal(torch.tensor([1., -1.]),
                                     torch.tensor([10., 10.])).to_event(1))
    mean = torch.matmul(design, w.unsqueeze(-1)).squeeze(-1)
    y = pyro.sample("y", dist.Bernoulli(logits=mean).to_event(1))
    return y


def sigmoid_example(design):
    n = design.shape[-2]
    random_effect_k = pyro.sample("k", dist.Gamma(2.*torch.ones(n), torch.tensor(2.)))
    random_effect_offset = pyro.sample("w2", dist.Normal(torch.tensor(0.), torch.ones(n)))
    w1 = pyro.sample("w1", dist.Normal(torch.tensor([1., -1.]),
                                       torch.tensor([10., 10.])).to_event(1))
    mean = torch.matmul(design[..., :-2], w1.unsqueeze(-1)).squeeze(-1)
    offset_mean = mean + random_effect_offset

    base_dist = dist.Normal(offset_mean, torch.tensor(1.)).to_event(1)
    transforms = [
        AffineTransform(loc=torch.tensor(0.), scale=random_effect_k),
        SigmoidTransform()
    ]
    response_dist = dist.TransformedDistribution(base_dist, transforms)
    y = pyro.sample("y", response_dist)
    return y


@pytest.mark.parametrize("model1,model2,design", [
    (
        zero_mean_unit_obs_sd_lm(torch.tensor([10., 10.]))[0],
        lm_2p_10_10_1,
        torch.tensor([[1., -1.]])
    ),
    (
        lm_2p_10_10_1,
        zero_mean_unit_obs_sd_lm(torch.tensor([10., 10.]))[0],
        torch.tensor([[100., -100.]])
    ),
    (
        group_linear_model(torch.tensor(0.), torch.tensor([10.]), torch.tensor(0.),
                           torch.tensor([10.]), torch.tensor(1.)),
        lm_2p_10_10_1_w12,
        torch.tensor([[-1.5, 0.5], [1.5, 0.]])
    ),
    (
        known_covariance_linear_model(torch.tensor([1., -1.]), torch.tensor([10., 10.]), torch.tensor(1.)),
        nz_lm_2p_10_10_1,
        torch.tensor([[-1., 0.5], [2.5, -2.]])
    ),
    (
        normal_inverse_gamma_linear_model(torch.tensor([1., -1.]), torch.tensor(.1),
                                          torch.tensor(2.), torch.tensor(2.)),
        normal_inv_gamma_2_2_10_10,
        torch.tensor([[1., -0.5], [1.5, 2.]])
    ),
    (
        logistic_regression_model(torch.tensor([1., -1.]), torch.tensor(10.)),
        lr_10_10,
        torch.tensor([[6., -1.5], [.5, 0.]])
    ),
    (
        sigmoid_model(torch.tensor([1., -1.]), torch.tensor([10., 10.]),
                      torch.tensor(0.), torch.tensor([1., 1.]),
                      torch.tensor(1.),
                      torch.tensor(2.), torch.tensor(2.), torch.eye(2)),
        sigmoid_example,
        torch.cat([torch.tensor([[1., 1.], [.5, -2.5]]), torch.eye(2)], dim=-1)
    )
])
def test_log_prob_matches(model1, model2, design):
    trace = poutine.trace(model1).get_trace(design)
    trace.compute_log_prob()
    ks = [k for k in trace.nodes.keys() if not k.startswith("_")]
    data = {l: trace.nodes[l]["value"] for l in ks}
    lp = {l: trace.nodes[l]["log_prob"] for l in ks}
    cond_model = pyro.condition(model2, data=data)
    cond_trace = poutine.trace(cond_model).get_trace(design)
    cond_trace.compute_log_prob()
    assert trace.nodes.keys() == cond_trace.nodes.keys()
    lp2 = {l: trace.nodes[l]["log_prob"] for l in ks}
    for l in lp.keys():
        assert_equal(lp[l], lp2[l])
