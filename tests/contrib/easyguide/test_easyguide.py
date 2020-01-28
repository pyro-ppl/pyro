# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import io
import warnings

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.easyguide import EasyGuide, easy_guide
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.initialization import init_to_mean, init_to_median
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings


# The model from tutorial/source/easyguide.ipynb
def model(batch, subsample, full_size):
    with ignore_jit_warnings():
        num_time_steps = len(batch)
    result = [None] * num_time_steps
    drift = pyro.sample("drift", dist.LogNormal(-1, 0.5))
    with pyro.plate("data", full_size, subsample=subsample):
        z = 0.
        for t in range(num_time_steps):
            z = pyro.sample("state_{}".format(t),
                            dist.Normal(z, drift))
            result[t] = pyro.sample("obs_{}".format(t), dist.Bernoulli(logits=z),
                                    obs=batch[t])
    return torch.stack(result)


def check_guide(guide):
    full_size = 50
    batch_size = 20
    num_time_steps = 8
    pyro.set_rng_seed(123456789)
    data = model([None] * num_time_steps, torch.arange(full_size), full_size)
    assert data.shape == (num_time_steps, full_size)

    pyro.get_param_store().clear()
    pyro.set_rng_seed(123456789)
    svi = SVI(model, guide, Adam({"lr": 0.02}), Trace_ELBO())
    for epoch in range(2):
        beg = 0
        while beg < full_size:
            end = min(full_size, beg + batch_size)
            subsample = torch.arange(beg, end)
            batch = data[:, beg:end]
            beg = end
            svi.step(batch, subsample, full_size=full_size)


@pytest.mark.parametrize("init_fn", [None, init_to_mean, init_to_median])
def test_delta_smoke(init_fn):

    @easy_guide(model)
    def guide(self, batch, subsample, full_size):
        self.map_estimate("drift")
        with self.plate("data", full_size, subsample=subsample):
            self.group(match="state_[0-9]*").map_estimate()

    if init_fn is not None:
        guide.init = init_fn

    check_guide(guide)


class PickleGuide(EasyGuide):
    def __init__(self, model):
        super().__init__(model)
        self.init = init_to_median

    def guide(self, batch, subsample, full_size):
        self.map_estimate("drift")
        with self.plate("data", full_size, subsample=subsample):
            self.group(match="state_[0-9]*").map_estimate()


def test_serialize():
    guide = PickleGuide(model)
    check_guide(guide)

    # Work around https://github.com/pytorch/pytorch/issues/27972
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        f = io.BytesIO()
        torch.save(guide, f)
        f.seek(0)
        actual = torch.load(f)

    assert type(actual) == type(guide)
    assert dir(actual) == dir(guide)
    check_guide(guide)
    check_guide(actual)


@pytest.mark.parametrize("init_fn", [None, init_to_mean, init_to_median])
def test_subsample_smoke(init_fn):
    rank = 2

    @easy_guide(model)
    def guide(self, batch, subsample, full_size):
        self.map_estimate("drift")
        group = self.group(match="state_[0-9]*")
        cov_diag = pyro.param("state_cov_diag",
                              lambda: torch.full(group.event_shape, 0.01),
                              constraint=constraints.positive)
        cov_factor = pyro.param("state_cov_factor",
                                lambda: torch.randn(group.event_shape + (rank,)) * 0.01)
        with self.plate("data", full_size, subsample=subsample):
            loc = pyro.param("state_loc",
                             lambda: torch.full((full_size,) + group.event_shape, 0.5),
                             event_dim=1)
            group.sample("states", dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag))

    if init_fn is not None:
        guide.init = init_fn

    check_guide(guide)


@pytest.mark.parametrize("init_fn", [None, init_to_mean, init_to_median])
def test_amortized_smoke(init_fn):
    rank = 2

    @easy_guide(model)
    def guide(self, batch, subsample, full_size):
        num_time_steps, batch_size = batch.shape
        self.map_estimate("drift")

        group = self.group(match="state_[0-9]*")
        cov_diag = pyro.param("state_cov_diag",
                              lambda: torch.full(group.event_shape, 0.01),
                              constraint=constraints.positive)
        cov_factor = pyro.param("state_cov_factor",
                                lambda: torch.randn(group.event_shape + (rank,)) * 0.01)

        if not hasattr(self, "nn"):
            self.nn = torch.nn.Linear(group.event_shape.numel(), group.event_shape.numel())
            self.nn.weight.data.fill_(1.0 / num_time_steps)
            self.nn.bias.data.fill_(-0.5)
        pyro.module("state_nn", self.nn)
        with self.plate("data", full_size, subsample=subsample):
            loc = self.nn(batch.t())
            group.sample("states", dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag))

    if init_fn is not None:
        guide.init = init_fn

    check_guide(guide)


def test_overlapping_plates_ok():

    def model(batch, subsample, full_size):
        # This is ok because the shared plate is left of the nonshared plate.
        with pyro.plate("shared", full_size, subsample=subsample, dim=-2):
            x = pyro.sample("x", dist.Normal(0, 1))
            with pyro.plate("nonshared", 2, dim=-1):
                y = pyro.sample("y", dist.Normal(0, 1))
            xy = x + y.sum(-1, keepdim=True)
            return pyro.sample("z", dist.Normal(xy, 1),
                               obs=batch)

    @easy_guide(model)
    def guide(self, batch, subsample, full_size):
        with self.plate("shared", full_size, subsample=subsample, dim=-2):
            group = self.group(match="x|y")
            loc = pyro.param("guide_loc",
                             torch.zeros((full_size, 1) + group.event_shape),
                             event_dim=1)
            scale = pyro.param("guide_scale",
                               torch.ones((full_size, 1) + group.event_shape),
                               constraint=constraints.positive,
                               event_dim=1)
            group.sample("xy", dist.Normal(loc, scale).to_event(1))

    # Generate data.
    full_size = 5
    batch_size = 2
    data = model(None, torch.arange(full_size), full_size)
    assert data.shape == (full_size, 1)

    # Train for one epoch.
    pyro.get_param_store().clear()
    svi = SVI(model, guide, Adam({"lr": 0.02}), Trace_ELBO())
    beg = 0
    while beg < full_size:
        end = min(full_size, beg + batch_size)
        subsample = torch.arange(beg, end)
        batch = data[beg:end]
        beg = end
        svi.step(batch, subsample, full_size=full_size)


def test_overlapping_plates_error():

    def model(batch, subsample, full_size):
        # This is an error because the shared plate is right of the nonshared plate.
        with pyro.plate("shared", full_size, subsample=subsample, dim=-1):
            x = pyro.sample("x", dist.Normal(0, 1))
            with pyro.plate("nonshared", 2, dim=-2):
                y = pyro.sample("y", dist.Normal(0, 1))
            xy = x + y.sum(-2)
            return pyro.sample("z", dist.Normal(xy, 1),
                               obs=batch)

    @easy_guide(model)
    def guide(self, batch, subsample, full_size):
        with self.plate("shared", full_size, subsample=subsample, dim=-1):
            group = self.group(match="x|y")
            loc = pyro.param("guide_loc",
                             torch.zeros((full_size,) + group.event_shape),
                             event_dim=1)
            scale = pyro.param("guide_scale",
                               torch.ones((full_size,) + group.event_shape),
                               constraint=constraints.positive,
                               event_dim=1)
            group.sample("xy", dist.Normal(loc, scale).to_event(1))

    # Generate data.
    full_size = 5
    batch_size = 2
    data = model(None, torch.arange(full_size), full_size)
    assert data.shape == (full_size,)

    # Train for one epoch.
    pyro.get_param_store().clear()
    svi = SVI(model, guide, Adam({"lr": 0.02}), Trace_ELBO())
    beg = 0
    end = min(full_size, beg + batch_size)
    subsample = torch.arange(beg, end)
    batch = data[beg:end]
    beg = end
    with pytest.raises(ValueError, match="Group expects all per-site plates"):
        svi.step(batch, subsample, full_size=full_size)
