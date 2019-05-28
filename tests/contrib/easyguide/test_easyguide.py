from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide.initialization import init_to_mean, init_to_median
from pyro.contrib.easyguide import easy_guide
from pyro.infer import SVI, Trace_ELBO
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
            result[t] = pyro.sample("obs_{}".format(t),
                                   dist.Bernoulli(logits=z),
                                   obs=batch[t])
    return torch.stack(result)


def check_guide(guide):
    full_size = 50
    batch_size = 20
    num_time_steps = 8
    pyro.set_rng_seed(123456789)
    data = model([None] * num_time_steps, torch.arange(full_size), full_size)
    assert data.shape == (num_time_steps, full_size)

    full_size = data.size(-1)
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
