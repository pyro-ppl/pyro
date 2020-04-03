# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, EnergyDistance
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from tests.common import assert_close


def model(data=None, num_samples=None):
    stability = pyro.sample("stability", dist.Uniform(0, 2))
    skew = pyro.sample("skew", dist.Uniform(-1, 1))
    scale = pyro.sample("scale", dist.LogNormal(0, 10))
    loc = pyro.sample("loc", dist.Normal(0, scale))
    with pyro.plate("plate", num_samples if data is None else len(data)):
        return pyro.sample("x", dist.Stable(stability, skew, scale, loc),
                           obs=data)


INITS = {"stability": 1.8, "skew": 0.0, "scale": 1.0, "loc": 0.0}


def init_loc_fn(site):
    return torch.tensor(INITS[site["name"]])


@pytest.mark.parametrize("stability", [1.9, 1.5, 1.0, 0.5])
def test_heavy_tail(stability):
    truth = {"stability": stability, "skew": 0.0, "scale": 1.0, "loc": 0.0}
    num_samples = 1000
    data = poutine.condition(model, data=truth)(num_samples=num_samples)
    min_stability = min(1.0, stability - 0.1)
    guide = AutoDelta(model, init_loc_fn=init_loc_fn)

    # Train the guide.
    optim = Adam({"lr": 0.1})
    energy = EnergyDistance(beta=min_stability, num_particles=8)
    num_steps = 1001
    svi = SVI(model, guide, optim, energy)
    for step in range(num_steps):
        loss = svi.step(data)
        if step % 50 == 0:
            logging.info("step {} loss = {:0.4g}".format(step, loss / data.numel()))

    # Evaluate.
    median = guide.median()
    for name, expected in truth.items():
        actual = median[name].item()
        logging.info("{}: {:0.3g} vs {:0.3g}".format(name, actual, expected))
        assert_close(actual, expected, atol=0.1)
