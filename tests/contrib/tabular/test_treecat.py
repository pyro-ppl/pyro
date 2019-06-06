from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.contrib.tabular.treecat import _dirmul_log_prob, find_center_of_tree
from tests.common import assert_close


@pytest.mark.parametrize('expected_vertex,edges', [
    (0, []),
    (1, [(0, 1)]),
    (0, [(0, 1), (0, 2)]),
    (1, [(0, 1), (1, 2)]),
    (2, [(0, 2), (1, 2)]),
    (2, [(0, 1), (1, 2), (2, 3)]),
    (1, [(0, 1), (1, 2), (1, 3)]),
    (2, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    (2, [(0, 2), (1, 2), (2, 3), (2, 4)]),
])
def test_find_center_of_tree(expected_vertex, edges):
    edges = torch.tensor(edges)
    v = find_center_of_tree(edges)
    assert v == expected_vertex


@pytest.mark.parametrize('alpha', [0.5, 0.1])
@pytest.mark.parametrize('counts_shape', [(6,), (5, 4), (4, 3, 2)])
def test_dirmul_log_prob(alpha, counts_shape):
    counts = torch.randn(counts_shape).exp()
    actual = _dirmul_log_prob(alpha, counts)
    expected = (alpha + counts).lgamma().sum(-1) - (1 + counts).lgamma().sum(-1)
    assert_close(actual, expected)
