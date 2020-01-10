# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro.distributions as dist


def test_categorical_shape():
    probs = torch.ones(3, 2) / 2
    d = dist.Categorical(probs)
    assert d.batch_shape == (3,)
    assert d.event_shape == ()
    assert d.shape() == (3,)
    assert d.sample().size() == d.shape()


def test_one_hot_categorical_shape():
    probs = torch.ones(3, 2) / 2
    d = dist.OneHotCategorical(probs)
    assert d.batch_shape == (3,)
    assert d.event_shape == (2,)
    assert d.shape() == (3, 2)
    assert d.sample().size() == d.shape()


def test_normal_shape():
    loc = torch.zeros(3, 2)
    scale = torch.ones(3, 2)
    d = dist.Normal(loc, scale)
    assert d.batch_shape == (3, 2)
    assert d.event_shape == ()
    assert d.shape() == (3, 2)
    assert d.sample().size() == d.shape()


def test_dirichlet_shape():
    alpha = torch.ones(3, 2) / 2
    d = dist.Dirichlet(alpha)
    assert d.batch_shape == (3,)
    assert d.event_shape == (2,)
    assert d.shape() == (3, 2)
    assert d.sample().size() == d.shape()


def test_zip_shape():
    gate = torch.ones(3, 2) / 2
    rate = torch.ones(3, 2) / 2
    d = dist.ZeroInflatedPoisson(gate, rate)
    assert d.batch_shape == (3, 2)
    assert d.event_shape == ()
    assert d.shape() == (3, 2)
    assert d.sample().size() == d.shape()


def test_bernoulli_log_prob_shape():
    probs = torch.ones(3, 2)
    x = torch.ones(3, 2)
    d = dist.Bernoulli(probs)
    assert d.log_prob(x).size() == (3, 2)


def test_categorical_log_prob_shape():
    probs = torch.ones(3, 2, 4) / 4
    x = torch.zeros(3, 2)
    d = dist.Categorical(probs)
    assert d.log_prob(x).size() == (3, 2)


def test_one_hot_categorical_log_prob_shape():
    probs = torch.ones(3, 2, 4) / 4
    x = torch.zeros(3, 2, 4)
    x[:, :, 0] = 1
    d = dist.OneHotCategorical(probs)
    assert d.log_prob(x).size() == (3, 2)


def test_normal_log_prob_shape():
    loc = torch.zeros(3, 2)
    scale = torch.ones(3, 2)
    x = torch.zeros(3, 2)
    d = dist.Normal(loc, scale)
    assert d.log_prob(x).size() == (3, 2)


def test_diag_normal_log_prob_shape():
    loc1 = torch.zeros(2, 3)
    loc2 = torch.zeros(2, 4)
    scale = torch.ones(2, 1)
    d1 = dist.Normal(loc1, scale.expand_as(loc1)).to_event(1)
    d2 = dist.Normal(loc2, scale.expand_as(loc2)).to_event(1)
    x1 = d1.sample()
    x2 = d2.sample()
    assert d1.log_prob(x1).size() == (2,)
    assert d2.log_prob(x2).size() == (2,)
