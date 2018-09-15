from __future__ import absolute_import, division, print_function

from collections import defaultdict

import pytest
import torch

import pyro
from pyro.contrib.tabular.trees import make_complete_graph, sample_tree
from tests.common import assert_equal

# https://oeis.org/A000272
NUM_SPANNING_TREES = [1, 1, 1, 3, 16, 125, 1296, 16807, 262144, 4782969]


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


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
def test_sample_tree_smoke(num_edges):
    pyro.set_rng_seed(num_edges)
    E = num_edges
    V = 1 + E
    grid = make_complete_graph(V)
    K = grid.shape[1]
    edge_logits = torch.rand(K)
    edges = [(v, v + 1) for v in range(V - 1)]
    for _ in range(100):
        edges = sample_tree(grid, edge_logits, edges)


@pytest.mark.parametrize('num_edges', [1, 2, 3, 4, 5])
def test_sample_tree_gof(num_edges):
    goftests = pytest.importorskip('goftests')
    pyro.set_rng_seed(num_edges)
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
