# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_close


def _hash(value):
    return tuple(value.tolist())


@pytest.mark.parametrize("num_destins", [1, 2, 3, 4, 5])
def test_enumerate(num_destins):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins)
    d = dist.OneTwoMatching(logits)
    values = d.enumerate_support()
    logging.info("destins = {}, suport size = {}".format(num_destins, len(values)))
    assert d.support.check(values), "invalid"
    assert len(set(map(_hash, values))) == len(values), "not unique"


@pytest.mark.parametrize("num_destins", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("bp_iters", [None, 10])
def test_log_prob(num_destins, bp_iters):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins)
    d = dist.OneTwoMatching(logits, bp_iters=bp_iters)
    values = d.enumerate_support()
    total = d.log_prob(values).exp().sum().item()
    assert_close(total, 1., atol=0.01)


@pytest.mark.parametrize("num_leaves", [3, 5, 8, 13, 100, 1000])
def test_log_prob_smoke(num_leaves):
    # Construct a random phylogenetic problem.
    leaf_times = torch.randn(num_leaves)
    coal_times = dist.CoalescentTimes(leaf_times).sample()
    times = torch.cat([leaf_times, coal_times]).requires_grad_()

    # Convert to a one-two-matching problem.
    ids = torch.arange(len(times))
    root = times.min(0).indices.item()
    sources = torch.cat([ids[:root], ids[root+1:]])
    destins = ids[num_leaves:]
    dt = times[sources][:, None] - times[destins]
    logits = torch.where(dt > 0, -dt, -math.inf)

    d = dist.OneTwoMatching(logits, bp_iters=10)
    logz = d.log_partition_function
    assert not torch.isnan(logz)
    dt = torch.autograd.grad(logz, [times])[0]
    assert not torch.isnan(dt).any()
