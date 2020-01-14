# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import Counter

import pytest
import torch

import pyro
from pyro.distributions.spanning_tree import NUM_SPANNING_TREES, SpanningTree, make_complete_graph, sample_tree
from tests.common import assert_equal, xfail_if_not_implemented

pytestmark = pytest.mark.skipif("CUDA_TEST" in os.environ, reason="spanning_tree unsupported on CUDA.")


@pytest.mark.filterwarnings("always")
@pytest.mark.parametrize('num_vertices,expected_grid', [
    (2, [[0], [1]]),
    (3, [[0, 0, 1], [1, 2, 2]]),
    (4, [[0, 0, 1, 0, 1, 2], [1, 2, 2, 3, 3, 3]]),
])
@pytest.mark.parametrize('backend', ["python", "cpp"])
def test_make_complete_graph(num_vertices, expected_grid, backend):
    V = num_vertices
    K = V * (V - 1) // 2
    expected_grid = torch.tensor(expected_grid, dtype=torch.long).reshape(2, K)

    grid = make_complete_graph(V, backend=backend)
    assert_equal(grid, expected_grid)


@pytest.mark.filterwarnings("always")
@pytest.mark.parametrize('num_edges', [1, 3, 10, 30, 100])
@pytest.mark.parametrize('backend', ["python", "cpp"])
def test_sample_tree_mcmc_smoke(num_edges, backend):
    pyro.set_rng_seed(num_edges)
    E = num_edges
    V = 1 + E
    K = V * (V - 1) // 2
    edge_logits = torch.rand(K)
    edges = torch.tensor([(v, v + 1) for v in range(V - 1)], dtype=torch.long)
    for _ in range(10 if backend == "cpp" or num_edges <= 30 else 1):
        edges = sample_tree(edge_logits, edges, backend=backend)


@pytest.mark.filterwarnings("always")
@pytest.mark.parametrize('num_edges', [1, 3, 10, 30, 100])
@pytest.mark.parametrize('backend', ["python", "cpp"])
def test_sample_tree_approx_smoke(num_edges, backend):
    pyro.set_rng_seed(num_edges)
    E = num_edges
    V = 1 + E
    K = V * (V - 1) // 2
    edge_logits = torch.rand(K)
    for _ in range(10 if backend == "cpp" or num_edges <= 30 else 1):
        sample_tree(edge_logits, backend=backend)


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5, 6])
def test_enumerate_support(num_edges):
    pyro.set_rng_seed(2 ** 32 - num_edges)
    E = num_edges
    V = 1 + E
    K = V * (V - 1) // 2
    edge_logits = torch.randn(K)
    d = SpanningTree(edge_logits)
    with xfail_if_not_implemented():
        support = d.enumerate_support()
    assert support.dim() == 3
    assert support.shape[1:] == d.event_shape
    assert support.size(0) == NUM_SPANNING_TREES[V]


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5, 6])
def test_partition_function(num_edges):
    pyro.set_rng_seed(2 ** 32 - num_edges)
    E = num_edges
    V = 1 + E
    K = V * (V - 1) // 2
    edge_logits = torch.randn(K)
    d = SpanningTree(edge_logits)
    with xfail_if_not_implemented():
        support = d.enumerate_support()
    v1 = support[..., 0]
    v2 = support[..., 1]
    k = v1 + v2 * (v2 - 1) // 2
    expected = edge_logits[k].sum(-1).logsumexp(0)
    actual = d.log_partition_function
    assert (actual - expected).abs() < 1e-6, (actual, expected)


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5, 6])
def test_log_prob(num_edges):
    pyro.set_rng_seed(2 ** 32 - num_edges)
    E = num_edges
    V = 1 + E
    K = V * (V - 1) // 2
    edge_logits = torch.randn(K)
    d = SpanningTree(edge_logits)
    with xfail_if_not_implemented():
        support = d.enumerate_support()
    log_probs = d.log_prob(support)
    assert log_probs.shape == (len(support),)
    log_total = log_probs.logsumexp(0).item()
    assert abs(log_total) < 1e-6, log_total


@pytest.mark.filterwarnings("always")
@pytest.mark.parametrize('pattern', ["uniform", "random", "sparse"])
@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('backend', ["python", "cpp"])
@pytest.mark.parametrize('method', ["mcmc", "approx"])
def test_sample_tree_gof(method, backend, num_edges, pattern):
    goftests = pytest.importorskip('goftests')
    pyro.set_rng_seed(2 ** 32 - num_edges)
    E = num_edges
    V = 1 + E
    K = V * (V - 1) // 2

    if pattern == "uniform":
        edge_logits = torch.zeros(K)
        num_samples = 10 * NUM_SPANNING_TREES[V]
    elif pattern == "random":
        edge_logits = torch.rand(K)
        num_samples = 30 * NUM_SPANNING_TREES[V]
    elif pattern == "sparse":
        edge_logits = torch.rand(K)
        for v2 in range(V):
            for v1 in range(v2):
                if v1 + 1 < v2:
                    edge_logits[v1 + v2 * (v2 - 1) // 2] = -float('inf')
        num_samples = 10 * NUM_SPANNING_TREES[V]

    # Generate many samples.
    counts = Counter()
    tensors = {}
    # Initialize using approximate sampler, to ensure feasibility.
    edges = sample_tree(edge_logits, backend=backend)
    for _ in range(num_samples):
        if method == "approx":
            # Reset the chain with an approximate sample, then perform 1 step of mcmc.
            edges = sample_tree(edge_logits, backend=backend)
        edges = sample_tree(edge_logits, edges, backend=backend)
        key = tuple((v1.item(), v2.item()) for v1, v2 in edges)
        counts[key] += 1
        tensors[key] = edges
    if pattern != "sparse":
        assert len(counts) == NUM_SPANNING_TREES[V]

    # Check accuracy using a Pearson's chi-squared test.
    keys = [k for k, _ in counts.most_common(100)]
    truncated = (len(keys) < len(counts))
    counts = torch.tensor([counts[k] for k in keys])
    tensors = torch.stack([tensors[k] for k in keys])
    probs = SpanningTree(edge_logits).log_prob(tensors).exp()
    gof = goftests.multinomial_goodness_of_fit(
        probs.numpy(), counts.numpy(), num_samples, plot=True, truncated=truncated)
    logging.info('gof = {}'.format(gof))
    if method == "approx":
        assert gof >= 0.0001
    else:
        assert gof >= 0.005
