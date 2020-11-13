# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_close

BP_ITERS = 30


def random_phylo_logits(num_leaves, dtype):
    # Construct a random phylogenetic problem.
    leaf_times = torch.randn(num_leaves, dtype=dtype)
    coal_times = dist.CoalescentTimes(leaf_times).sample()
    times = torch.cat([leaf_times, coal_times]).requires_grad_()
    assert times.dtype == dtype

    # Convert to a one-two-matching problem.
    ids = torch.arange(len(times))
    root = times.min(0).indices.item()
    sources = torch.cat([ids[:root], ids[root+1:]])
    destins = ids[num_leaves:]
    dt = times[sources][:, None] - times[destins]
    dt = dt * 10 / dt.detach().std()
    logits = torch.where(dt > 0, -dt, dt.new_tensor(-math.inf))
    assert logits.dtype == dtype

    return logits, times


def _hash(value):
    return tuple(value.tolist())


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_destins", [1, 2, 3, 4, 5])
def test_enumerate(num_destins, dtype):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins, dtype=dtype)
    d = dist.OneTwoMatching(logits)
    values = d.enumerate_support()
    logging.info("destins = {}, suport size = {}".format(num_destins, len(values)))
    assert d.support.check(values), "invalid"
    assert len(set(map(_hash, values))) == len(values), "not unique"


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("num_destins", [1, 2, 3, 4, 5])
def test_sample_smoke(num_destins, sample_shape, dtype):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins, dtype=dtype)
    d = dist.OneTwoMatching(logits)
    values = d.sample(sample_shape)
    assert values.shape == sample_shape + (num_sources,)
    assert d.support.check(values).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_destins", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_log_prob_full(num_destins, dtype, bp_iters):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins, dtype=dtype) * 10
    d = dist.OneTwoMatching(logits, bp_iters=bp_iters)
    values = d.enumerate_support()
    log_total = d.log_prob(values).logsumexp(0).item()
    logging.info(f"log_total = {log_total:0.3g}, " +
                 f"log_Z = {d.log_partition_function:0.3g}")
    assert_close(log_total, 0., atol=min(num_destins, 1.0))


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_log_prob_hard(dtype, bp_iters):
    logits = [[0., 0.], [0., 0.], [0., 0.], [0., -math.inf]]
    logits = torch.tensor(logits, dtype=dtype)
    d = dist.OneTwoMatching(logits, bp_iters=bp_iters)
    values = d.enumerate_support()
    log_total = d.log_prob(values).logsumexp(0).item()
    logging.info(f"log_total = {log_total:0.3g}, " +
                 f"log_Z = {d.log_partition_function:0.3g}")
    assert_close(log_total, 0., atol=0.5)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_log_prob_phylo(num_leaves, dtype, bp_iters):
    logits, times = random_phylo_logits(num_leaves, dtype)
    d = dist.OneTwoMatching(logits, bp_iters=bp_iters)
    values = d.enumerate_support()
    log_total = d.log_prob(values).logsumexp(0).item()
    logging.info(f"log_total = {log_total:0.3g}, " +
                 f"log_Z = {d.log_partition_function:0.3g}")
    assert_close(log_total, 0., atol=2.0)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_leaves", [3, 5, 8, 13, 100, 1000])
def test_log_prob_phylo_smoke(num_leaves, dtype):
    logits, times = random_phylo_logits(num_leaves, dtype)
    d = dist.OneTwoMatching(logits, bp_iters=10)
    logz = d.log_partition_function
    assert logz.dtype == dtype
    assert not torch.isnan(logz)
    dt = torch.autograd.grad(logz, [times])[0]
    assert not torch.isnan(dt).any()
