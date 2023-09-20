# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_close, assert_equal, xfail_if_not_implemented

BP_ITERS = 50


def _hash(value):
    return tuple(value.tolist())


def random_phylo_logits(num_leaves, dtype):
    # Construct a random phylogenetic problem.
    leaf_times = torch.randn(num_leaves, dtype=dtype)
    coal_times = dist.CoalescentTimes(leaf_times).sample()
    times = torch.cat([leaf_times, coal_times]).requires_grad_()
    assert times.dtype == dtype

    # Convert to a one-two-matching problem.
    ids = torch.arange(len(times))
    root = times.min(0).indices.item()
    sources = torch.cat([ids[:root], ids[root + 1 :]])
    destins = ids[num_leaves:]
    dt = times[sources][:, None] - times[destins]
    dt = dt * 10 / dt.detach().std()
    logits = torch.where(dt > 0, -dt, dt.new_tensor(-math.inf))
    assert logits.dtype == dtype
    logits.data += torch.empty_like(logits).uniform_()  # add jitter

    return logits, times


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
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_sample_shape_smoke(num_destins, sample_shape, dtype, bp_iters):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins, dtype=dtype)
    d = dist.OneTwoMatching(logits, bp_iters=bp_iters)
    with xfail_if_not_implemented():
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
    logging.info(
        f"log_total = {log_total:0.3g}, " + f"log_Z = {d.log_partition_function:0.3g}"
    )
    assert_close(log_total, 0.0, atol=1.0)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_log_prob_hard(dtype, bp_iters):
    logits = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, -math.inf]]
    logits = torch.tensor(logits, dtype=dtype)
    d = dist.OneTwoMatching(logits, bp_iters=bp_iters)
    values = d.enumerate_support()
    log_total = d.log_prob(values).logsumexp(0).item()
    logging.info(
        f"log_total = {log_total:0.3g}, " + f"log_Z = {d.log_partition_function:0.3g}"
    )
    assert_close(log_total, 0.0, atol=0.5)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_log_prob_phylo(num_leaves, dtype, bp_iters):
    logits, times = random_phylo_logits(num_leaves, dtype)
    d = dist.OneTwoMatching(logits, bp_iters=bp_iters)
    values = d.enumerate_support()
    log_total = d.log_prob(values).logsumexp(0).item()
    logging.info(
        f"log_total = {log_total:0.3g}, " + f"log_Z = {d.log_partition_function:0.3g}"
    )
    assert_close(log_total, 0.0, atol=1.0)


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


def assert_grads_ok(logits, bp_iters=None):
    def fn(logits):
        d = dist.OneTwoMatching(logits, bp_iters=bp_iters)
        return d.log_partition_function

    torch.autograd.gradcheck(fn, logits, atol=1e-3, rtol=1e-3)


def assert_grads_agree(logits):
    d1 = dist.OneTwoMatching(logits)
    d2 = dist.OneTwoMatching(logits, bp_iters=BP_ITERS)
    expected = torch.autograd.grad(d1.log_partition_function, [logits])[0]
    actual = torch.autograd.grad(d2.log_partition_function, [logits])[0]
    assert torch.allclose(
        actual, expected, atol=0.2, rtol=1e-3
    ), f"Expected:\n{expected.numpy()}\nActual:\n{actual.numpy()}"


@pytest.mark.parametrize("num_destins", [2, 3, 4, 5])
def test_grad_full(num_destins):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins) * 10
    logits.requires_grad_()

    assert_grads_ok(logits)
    assert_grads_ok(logits, bp_iters=BP_ITERS)
    assert_grads_agree(logits)


@pytest.mark.parametrize("num_destins", [2, 3, 4])
def test_grad_hard(num_destins):
    num_sources = 2 * num_destins
    i = torch.arange(num_sources)[:, None]
    j = torch.arange(num_destins)
    logits = torch.randn(num_sources, num_destins) * 10
    logits[i < j] = -100
    logits.requires_grad_()

    assert_grads_ok(logits)
    assert_grads_ok(logits, bp_iters=BP_ITERS)
    assert_grads_agree(logits)


@pytest.mark.parametrize("num_leaves", [2, 3, 4, 5])
def test_grad_phylo(num_leaves):
    logits, times = random_phylo_logits(num_leaves, torch.double)
    logits = logits.detach().requires_grad_()

    assert_grads_ok(logits)
    assert_grads_ok(logits, bp_iters=BP_ITERS)
    assert_grads_agree(logits)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_destins", [1, 2, 3, 4, 5])
def test_mode_full(num_destins, dtype):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins, dtype=dtype) * 10
    d = dist.OneTwoMatching(logits)
    values = d.enumerate_support()
    i = d.log_prob(values).max(0).indices.item()
    expected = values[i]
    actual = d.mode()
    assert_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 3, 4, 5, 6])
def test_mode_phylo(num_leaves, dtype):
    logits, times = random_phylo_logits(num_leaves, dtype)
    d = dist.OneTwoMatching(logits)
    values = d.enumerate_support()
    i = d.log_prob(values).max(0).indices.item()
    expected = values[i]
    actual = d.mode()
    assert_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_destins", [3, 5, 8, 13, 100, 1000])
def test_mode_full_smoke(num_destins, dtype):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins, dtype=dtype) * 10
    d = dist.OneTwoMatching(logits)
    value = d.mode()
    assert d.support.check(value)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_leaves", [3, 5, 8, 13, 100, 1000])
def test_mode_phylo_smoke(num_leaves, dtype):
    logits, times = random_phylo_logits(num_leaves, dtype)
    d = dist.OneTwoMatching(logits, bp_iters=10)
    value = d.mode()
    assert d.support.check(value)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_destins", [2, 3, 4])
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_sample_full(num_destins, dtype, bp_iters):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins, dtype=dtype) * 10
    d = dist.OneTwoMatching(logits, bp_iters=bp_iters)

    # Compute an empirical mean.
    num_samples = 1000
    s = torch.arange(num_sources)
    actual = torch.zeros_like(logits)
    with xfail_if_not_implemented():
        for v in d.sample([num_samples]):
            actual[s, v] += 1 / num_samples

    # Compute truth via enumeration.
    values = d.enumerate_support()
    probs = d.log_prob(values).exp()
    probs /= probs.sum()
    expected = torch.zeros(num_sources, num_destins)
    for v, p in zip(values, probs):
        expected[s, v] += p
    assert_close(actual, expected, atol=0.1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_leaves", [3, 4, 5])
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_sample_phylo(num_leaves, dtype, bp_iters):
    logits, times = random_phylo_logits(num_leaves, dtype)
    num_sources, num_destins = logits.shape
    d = dist.OneTwoMatching(logits, bp_iters=bp_iters)

    # Compute an empirical mean.
    num_samples = 1000
    s = torch.arange(num_sources)
    actual = torch.zeros_like(logits)
    with xfail_if_not_implemented():
        for v in d.sample([num_samples]):
            actual[s, v] += 1 / num_samples

    # Compute truth via enumeration.
    values = d.enumerate_support()
    probs = d.log_prob(values).exp()
    probs /= probs.sum()
    expected = torch.zeros(num_sources, num_destins)
    for v, p in zip(values, probs):
        expected[s, v] += p
    assert_close(actual, expected, atol=0.1)
