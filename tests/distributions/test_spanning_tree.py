from __future__ import absolute_import, division, print_function

from collections import defaultdict

import pytest
import torch

import pyro
from pyro.distributions.spanning_tree import (NUM_SPANNING_TREES, SpanningTree, make_complete_graph, sample_tree,
                                              sample_tree_2, sample_tree_3)
from tests.common import assert_equal, xfail_if_not_implemented


@pytest.mark.parametrize('num_vertices,expected_grid', [
    (2, [[0], [0], [1]]),
    (3, [[0, 1, 2], [0, 0, 1], [1, 2, 2]]),
    (4, [[0, 1, 2, 3, 4, 5], [0, 0, 1, 0, 1, 2], [1, 2, 2, 3, 3, 3]]),
])
def test_make_complete_graph(num_vertices, expected_grid):
    V = num_vertices
    K = V * (V - 1) // 2
    expected_grid = torch.tensor(expected_grid, dtype=torch.long).reshape(3, K)

    grid = make_complete_graph(V)
    assert_equal(grid, expected_grid)


@pytest.mark.parametrize('num_edges', [1, 3, 10, 30, 100])
def test_sample_tree_smoke(num_edges):
    pyro.set_rng_seed(num_edges)
    E = num_edges
    V = 1 + E
    grid = make_complete_graph(V)
    K = grid.shape[1]
    edge_logits = torch.rand(K)
    edges = [(v, v + 1) for v in range(V - 1)]
    for _ in range(10):
        edges = sample_tree(grid, edge_logits, edges)


@pytest.mark.parametrize('num_edges', [1, 3, 10, 30, 100])
def test_sample_tree_2_smoke(num_edges):
    pyro.set_rng_seed(num_edges)
    E = num_edges
    V = 1 + E
    grid = make_complete_graph(V)
    K = grid.shape[1]
    edge_logits = torch.rand(K)
    for _ in range(10):
        sample_tree_2(grid, edge_logits)


@pytest.mark.parametrize('num_edges', [1, 3, 10, 30, 100])
def test_sample_tree_3_smoke(num_edges):
    pyro.set_rng_seed(num_edges)
    E = num_edges
    V = 1 + E
    K = V * (V - 1) // 2
    edge_logits = torch.rand(K)
    for _ in range(10):
        sample_tree_3(edge_logits)


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
def test_sample_tree_gof(num_edges):
    goftests = pytest.importorskip('goftests')
    pyro.set_rng_seed(2 ** 32 - num_edges)
    E = num_edges
    V = 1 + E
    grid = make_complete_graph(V)
    K = grid.shape[1]
    edge_logits = torch.rand(K)
    edge_logits_dict = {(v1, v2): edge_logits[k] for k, v1, v2 in grid.t().numpy()}

    # Generate many samples via MCMC.
    num_samples = 30 * NUM_SPANNING_TREES[V]
    counts = defaultdict(int)
    edges = [(v, v + 1) for v in range(V - 1)]
    for _ in range(num_samples):
        edges = sample_tree(grid, edge_logits, edges)
        counts[tuple(edges)] += 1
    assert len(counts) == NUM_SPANNING_TREES[V]

    # Check accuracy using Pearson's chi-squared test.
    keys = counts.keys()
    counts = torch.tensor([counts[key] for key in keys])
    probs = torch.tensor([sum(edge_logits_dict[edge] for edge in key) for key in keys])
    probs /= probs.sum()

    # Possibly truncate.
    T = 100
    truncated = False
    if len(counts) > T:
        counts = counts[:T]
        probs = probs[:T]
        truncated = True

    gof = goftests.multinomial_goodness_of_fit(
        probs.numpy(), counts.numpy(), num_samples, plot=True, truncated=truncated)
    assert 1e-2 < gof


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
def test_sample_tree_2_gof(num_edges):
    goftests = pytest.importorskip('goftests')
    pyro.set_rng_seed(2 ** 32 - num_edges)
    E = num_edges
    V = 1 + E
    grid = make_complete_graph(V)
    K = grid.shape[1]
    edge_logits = torch.rand(K)
    edge_logits_dict = {(v1, v2): edge_logits[k] for k, v1, v2 in grid.t().numpy()}

    # Generate many samples.
    num_samples = 30 * NUM_SPANNING_TREES[V]
    counts = defaultdict(int)
    for _ in range(num_samples):
        edges = sample_tree_2(grid, edge_logits)
        counts[edges] += 1
    assert len(counts) == NUM_SPANNING_TREES[V]

    # Check accuracy using Pearson's chi-squared test.
    keys = counts.keys()
    counts = torch.tensor([counts[key] for key in keys])
    probs = torch.tensor([sum(edge_logits_dict[edge] for edge in key) for key in keys])
    probs /= probs.sum()

    # Possibly truncate.
    T = 100
    truncated = False
    if len(counts) > T:
        counts = counts[:T]
        probs = probs[:T]
        truncated = True

    gof = goftests.multinomial_goodness_of_fit(
        probs.numpy(), counts.numpy(), num_samples, plot=True, truncated=truncated)
    assert 1e-2 < gof


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
def test_sample_tree_3_gof(num_edges):
    goftests = pytest.importorskip('goftests')
    pyro.set_rng_seed(2 ** 32 - num_edges)
    E = num_edges
    V = 1 + E
    grid = make_complete_graph(V)
    K = grid.shape[1]
    edge_logits = torch.rand(K)
    edge_logits_dict = {(v1, v2): edge_logits[k] for k, v1, v2 in grid.t().numpy()}

    # Generate many samples.
    num_samples = 30 * NUM_SPANNING_TREES[V]
    counts = defaultdict(int)
    for _ in range(num_samples):
        edges = sample_tree_3(edge_logits)
        edges = tuple(sorted((v1.item(), v2.item()) for v1, v2 in edges))
        counts[edges] += 1
    assert len(counts) == NUM_SPANNING_TREES[V]

    # Check accuracy using Pearson's chi-squared test.
    keys = counts.keys()
    counts = torch.tensor([counts[key] for key in keys])
    probs = torch.tensor([sum(edge_logits_dict[edge] for edge in key) for key in keys])
    probs /= probs.sum()

    # Possibly truncate.
    T = 100
    truncated = False
    if len(counts) > T:
        counts = counts[:T]
        probs = probs[:T]
        truncated = True

    gof = goftests.multinomial_goodness_of_fit(
        probs.numpy(), counts.numpy(), num_samples, plot=True, truncated=truncated)
    assert 1e-2 < gof


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
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


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
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


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
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
