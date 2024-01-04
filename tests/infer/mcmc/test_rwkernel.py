# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.rwkernel import RandomWalkKernel
from tests.common import assert_equal


def test_beta_bernoulli():
    alpha = torch.tensor([1.1, 2.2])
    beta = torch.tensor([1.1, 2.2])

    def model(data):
        p_latent = pyro.sample("p_latent", dist.Beta(alpha, beta))
        with pyro.plate("data", data.shape[0], dim=-2):
            pyro.sample("obs", dist.Bernoulli(p_latent), obs=data)

    num_data = 5
    true_probs = torch.tensor([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((num_data,))))

    kernel = RandomWalkKernel(model)
    mcmc = MCMC(kernel, num_samples=2000, warmup_steps=500)
    mcmc.run(data)
    samples = mcmc.get_samples()

    data_sum = data.sum(0)
    alpha_post = alpha + data_sum
    beta_post = beta + num_data - data_sum
    expected_mean = alpha_post / (alpha_post + beta_post)
    expected_var = (
        expected_mean.pow(2) * beta_post / (alpha_post * (1 + alpha_post + beta_post))
    )

    assert_equal(samples["p_latent"].mean(0), expected_mean, prec=0.03)
    assert_equal(samples["p_latent"].var(0), expected_var, prec=0.005)
