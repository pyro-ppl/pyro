from __future__ import absolute_import, division, print_function

import logging
import os
from collections import namedtuple

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC
import pyro.poutine as poutine
from tests.common import assert_equal


def bernoulli_latent_model():
    @poutine.broadcast
    def model(data):
        y_prob = pyro.sample("y_prob", dist.Beta(1.0, 1.0))
        y = pyro.sample("y", dist.Bernoulli(y_prob))
        with pyro.iarange("data", data.shape[0]):
            z = pyro.sample("z", dist.Bernoulli(0.65 * y + 0.1))
            pyro.sample("obs", dist.Normal(2. * z, 1.), obs=data)
        pyro.sample("nuisance", dist.Bernoulli(0.3))

    N = 2000
    y_prob = torch.tensor(0.3)
    y = dist.Bernoulli(y_prob).sample(torch.Size((N,)))
    z = dist.Bernoulli(0.65 * y + 0.1).sample()
    data = dist.Normal(2. * z, 1.0).sample()
    hmc_kernel = HMC(model, trajectory_length=1, adapt_step_size=True, max_iarange_nesting=1)
    mcmc_run = MCMC(hmc_kernel, num_samples=600, warmup_steps=200).run(data)


bernoulli_latent_model()