import pytest

from pyro.poutine import Trace
from tests.common import assert_equal


@pytest.mark.parametrize("root,edges", [
    #   1
    #  / \
    # 2   3
    #    / \
    #   4   5
    #  / \
    # 6   7
    (1, [(1, 2), (1, 3), (3, 4), (3, 5), (4, 6), (4, 7)]),
    #   1
    #  /|\
    # 2 3 4
    #   |
    #   5
    #   |
    #   6
    #   |
    #   7
    (1, [(1, 2), (3, 5), (1, 4), (1, 3), (5, 6), (6, 7)]),
])
def test_top_sort(root, edges):
    tr = Trace()
    for e1, e2 in edges:
        tr.add_edge(e1, e2)
    top_sort = tr.topsort()
    while top_sort:
        num_nodes = len([n for n in tr._dfs(root, set())])
        num_expected = len(top_sort)
        assert_equal(num_nodes, num_expected)
        tr.remove_node(top_sort.pop())
