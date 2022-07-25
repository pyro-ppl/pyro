# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.resampler import Resampler
from tests.common import assert_close


@pytest.mark.parametrize("stable", [False, True])
def test_resampling_cache(stable):
    def model_(a):
        pyro.sample("alpha", dist.Dirichlet(a))

    def simulator():
        a = torch.tensor([2.0, 1.0, 1.0, 2.0])
        alpha = pyro.sample("alpha", dist.Dirichlet(a))
        pyro.sample("x", dist.Normal(alpha, 0.01).to_event(1))

    # initialize
    a = torch.tensor([1.0, 2.0, 1.0, 1.0])
    guide = functools.partial(model_, a)
    resampler = Resampler(guide, simulator, num_guide_samples=10000)

    # resample
    b = torch.tensor([1.0, 2.0, 3.0, 4.0])
    model = functools.partial(model_, b)
    samples = resampler.sample(model, 1000)
    assert all(v.shape[:1] == (1000,) for v in samples.values())
    num_unique = len(set(map(tuple, samples["alpha"].tolist())))
    assert num_unique >= 500

    # check moments
    expected_mean = b / b.sum()
    actual_mean = samples["alpha"].mean(0)
    assert_close(actual_mean, expected_mean, atol=0.01)
    actual_mean = samples["x"].mean(0)
    assert_close(actual_mean, expected_mean, atol=0.01)
