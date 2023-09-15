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


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_nodes", [1, 2, 3, 4, 5, 6])
def test_enumerate(num_nodes, dtype):
    logits = torch.randn(num_nodes, num_nodes, dtype=dtype)
    d = dist.OneOneMatching(logits)
    values = d.enumerate_support()
    logging.info("destins = {}, suport size = {}".format(num_nodes, len(values)))
    assert d.support.check(values), "invalid"
    assert len(set(map(_hash, values))) == len(values), "not unique"


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("num_nodes", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_sample_shape_smoke(num_nodes, sample_shape, dtype, bp_iters):
    logits = torch.randn(num_nodes, num_nodes, dtype=dtype)
    d = dist.OneOneMatching(logits, bp_iters=bp_iters)
    with xfail_if_not_implemented():
        values = d.sample(sample_shape)
    assert values.shape == sample_shape + (num_nodes,)
    assert d.support.check(values).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_nodes", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_log_prob_full(num_nodes, dtype, bp_iters):
    logits = torch.randn(num_nodes, num_nodes, dtype=dtype) * 10
    d = dist.OneOneMatching(logits, bp_iters=bp_iters)
    values = d.enumerate_support()
    log_total = d.log_prob(values).logsumexp(0).item()
    logging.info(
        f"log_total = {log_total:0.3g}, " + f"log_Z = {d.log_partition_function:0.3g}"
    )
    assert_close(log_total, 0.0, atol=2.0)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_log_prob_hard(dtype, bp_iters):
    logits = [[0.0, 0.0], [0.0, -math.inf]]
    logits = torch.tensor(logits, dtype=dtype)
    d = dist.OneOneMatching(logits, bp_iters=bp_iters)
    values = d.enumerate_support()
    log_total = d.log_prob(values).logsumexp(0).item()
    logging.info(
        f"log_total = {log_total:0.3g}, " + f"log_Z = {d.log_partition_function:0.3g}"
    )
    assert_close(log_total, 0.0, atol=0.5)


def assert_grads_ok(logits, bp_iters=None):
    def fn(logits):
        d = dist.OneOneMatching(logits, bp_iters=bp_iters)
        return d.log_partition_function

    torch.autograd.gradcheck(fn, logits, atol=1e-3, rtol=1e-3)


def assert_grads_agree(logits):
    d1 = dist.OneOneMatching(logits)
    d2 = dist.OneOneMatching(logits, bp_iters=BP_ITERS)
    expected = torch.autograd.grad(d1.log_partition_function, [logits])[0]
    actual = torch.autograd.grad(d2.log_partition_function, [logits])[0]
    assert torch.allclose(
        actual, expected, atol=0.2, rtol=1e-3
    ), f"Expected:\n{expected.numpy()}\nActual:\n{actual.numpy()}"


@pytest.mark.parametrize("num_nodes", [2, 3, 4, 5])
def test_grad_full(num_nodes):
    logits = torch.randn(num_nodes, num_nodes) * 10
    logits.requires_grad_()

    assert_grads_ok(logits)
    assert_grads_ok(logits, bp_iters=BP_ITERS)
    assert_grads_agree(logits)


@pytest.mark.parametrize("num_nodes", [2, 3, 4, 5])
def test_grad_hard(num_nodes):
    i = torch.arange(num_nodes)
    logits = torch.randn(num_nodes, num_nodes) * 10
    logits[i[:, None] < i] = -100
    logits.requires_grad_()

    assert_grads_ok(logits)
    assert_grads_ok(logits, bp_iters=BP_ITERS)
    assert_grads_agree(logits)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_nodes", [1, 2, 3, 4, 5, 6, 7, 8])
def test_mode(num_nodes, dtype):
    logits = torch.randn(num_nodes, num_nodes, dtype=dtype) * 10
    d = dist.OneOneMatching(logits)
    values = d.enumerate_support()
    i = d.log_prob(values).max(0).indices.item()
    expected = values[i]
    actual = d.mode()
    assert_equal(actual, expected)
    assert (actual == expected).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_nodes", [3, 5, 8, 13, 100, 1000])
def test_mode_smoke(num_nodes, dtype):
    logits = torch.randn(num_nodes, num_nodes, dtype=dtype) * 10
    d = dist.OneOneMatching(logits)
    value = d.mode()
    assert d.support.check(value)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
@pytest.mark.parametrize("num_nodes", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("bp_iters", [None, BP_ITERS], ids=["exact", "bp"])
def test_sample(num_nodes, dtype, bp_iters):
    logits = torch.randn(num_nodes, num_nodes, dtype=dtype) * 10
    d = dist.OneOneMatching(logits, bp_iters=bp_iters)

    # Compute an empirical mean.
    num_samples = 1000
    s = torch.arange(num_nodes)
    actual = torch.zeros_like(logits)
    with xfail_if_not_implemented():
        for v in d.sample([num_samples]):
            actual[s, v] += 1 / num_samples

    # Compute truth via enumeration.
    values = d.enumerate_support()
    probs = d.log_prob(values).exp()
    probs /= probs.sum()
    expected = torch.zeros(num_nodes, num_nodes)
    for v, p in zip(values, probs):
        expected[s, v] += p
    assert_close(actual, expected, atol=0.1)
