# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.resampler import ResamplingCache
from tests.common import assert_close


def test_resampling_cache():
    size = 4

    def prior(a):
        alpha = pyro.sample("alpha", dist.Dirichlet(a))

    def model():
        alpha = pyro.sample("alpha", dist.Dirichlet(3 * torch.ones(size)))
        x = pyro.sample("x", dist.Normal(alpha, 0.01).to_event(1))

    cache = ResamplingCache(model)

    num_steps = 3
    num_samples = 10000
    for _ in range(num_steps):
        a = 1 + torch.randn(size).exp()
        prior_a = functools.partial(prior, a=a)
        samples = cache.sample(prior_a, num_samples)

        # check moments
        expected_mean = a / a.sum()
        probs = samples["_weight"] / samples["_weight"].sum()
        actual_mean = probs @ samples["x"]
        assert_close(actual_mean, expected_mean, atol=0.01)
        print("cache size =", len(cache._cache["_logp"]))

    assert len(cache.cache) < num_steps * num_samples, "no sharing"
