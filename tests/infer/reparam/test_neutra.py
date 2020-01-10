# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, NUTS, MCMC
from pyro.infer.autoguide import AutoIAFNormal
from pyro import optim
from pyro.infer.reparam import NeuTraReparam


def test_neals_funnel_smoke():
    dim = 10

    def model():
        y = pyro.sample('y', dist.Normal(0, 3))
        with pyro.plate("D", dim):
            pyro.sample('x', dist.Normal(0, torch.exp(y/2)))

    guide = AutoIAFNormal(model)
    svi = SVI(model, guide,  optim.Adam({"lr": 1e-10}), Trace_ELBO())
    for _ in range(1000):
        svi.step()

    neutra = NeuTraReparam(guide)
    model = neutra.reparam(model)
    nuts = NUTS(model)
    mcmc = MCMC(nuts, num_samples=50, warmup_steps=50)
    mcmc.run()
    samples = mcmc.get_samples()
    # XXX: `MCMC.get_samples` adds a leftmost batch dim to all sites, not uniformly at -max_plate_nesting-1;
    # hence the unsqueeze
    transformed_samples = neutra.transform_sample(samples['y_shared_latent'].unsqueeze(-2))
    assert 'x' in transformed_samples
    assert 'y' in transformed_samples
