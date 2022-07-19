# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributions as dist

from pyro.infer.resampler import ResamplingCache
from tests.common import assert_close


@pytest.mark.parametrize("batch_size", [None, 10000])
def test_resampling_cache(batch_size):
    size = 4
    loc = torch.arange(float(size))

    def make_prior():
        alpha = dist.Gamma(2, torch.ones(size)).sample()
        return {"concentration": dist.Dirichlet(alpha)}

    def model(concentration):
        x = dist.Normal(loc, 0.1).sample(concentration.shape[:-1])
        return (x * concentration).sum(-1)

    cache = ResamplingCache(model, batch_size=batch_size)

    num_steps = 3
    num_samples = 10 * (1 if batch_size is None else batch_size)
    for _ in range(num_steps):
        prior = make_prior()
        samples = cache.sample(prior, num_samples)
        if batch_size:
            expected_mean = loc @ prior["concentration"].mean
            actual_mean = torch.stack(samples).mean(0)
            assert_close(actual_mean, expected_mean, atol=0.01)
            print(len(cache.cache))

    assert len(cache.cache) < num_steps * num_samples, "no sharing"
