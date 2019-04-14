from __future__ import absolute_import, division, print_function

import itertools

import torch
from six.moves import range
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch_distribution import TorchDistribution


class SpanningTree(TorchDistribution):
    """
    Distribution over spanning trees on a fixed number of vertices.
    """
    arg_constraints = {'edge_logits': constraints.real}
    support = constraints.positive_integer
    has_enumerate_support = True

    def __init__(self, edge_logits, initial_edges=None, mcmc_steps=1, validate_args=None):
        K = len(edge_logits)
        V = int(round(0.5 + (0.25 + 2 * K)**0.5))
        assert K == V * (V - 1) // 2
        E = V - 1
        event_shape = (E, 2)
        batch_shape = ()
        if initial_edges is None:
            initial_edges = torch.stack((torch.arange(0, V - 1), torch.arange(1, V)), dim=-1)
        self.edge_logits = edge_logits
        self.initial_edges = initial_edges
        super(SpanningTree, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        if self._validate_args:
            if edge_logits.shape != (K,):
                raise ValueError("Expected edge_logits of shape ({},), but got shape {}"
                                 .format(K, edge_logits.shape))
            if initial_edges.shape != (E, 2):
                raise ValueError("Expected initial_edges of shape ({},2), but got shape {}"
                                 .format(K, edge_logits.shape))
        self.mcmc_steps = mcmc_steps
        self.complete_graph = make_complete_graph(V)
        self.num_vertices = V

    @lazy_property
    def log_partition_function(self):
        # By Kirchoff's matrix-tree theorem, the partition function is the
        # determinant of a truncated version of the graph Laplacian matrix. We
        # use a Cholesky decomposition to compute the log determinant.
        # See https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem
        V = self.num_vertices
        grid = self.complete_graph
        shift = self.edge_logits.max()
        edge_probs = (self.edge_logits - shift).exp()
        adjacency = edge_probs.new_zeros(V, V)
        adjacency[grid[1], grid[2]] = edge_probs
        adjacency[grid[2], grid[1]] = edge_probs
        laplacian = adjacency.sum(-1).diag() - adjacency
        truncated = laplacian[:-1, :-1]
        log_det = torch.cholesky(truncated).diag().log().sum() * 2
        return log_det + shift * (V - 1)

    def log_prob(self, edges):
        if self._validate_args:
            assert edges.dim() >= 2
            assert edges.shape[-2:] == self.event_shape
        v1 = edges[..., 0]
        v2 = edges[..., 1]
        if self._validate_args:
            assert (v1 < v2).all()
        k = v1 + v2 * (v2 - 1) // 2
        return self.edge_logits[k].sum(-1) - self.log_partition_function

    def sample(self, sample_shape=torch.Size()):
        if sample_shape:
            raise NotImplementedError("SpanningTree does not support batching")
        edges = [(v1.item(), v2.item()) for v1, v2 in self.initial_edges]
        for _ in range(self.mcmc_steps):
            edges = sample_tree(self.complete_graph, self.edge_logits, edges)
        result = self.edge_logits.new_empty((len(edges), 2), dtype=torch.long)
        for e, vs in edges:
            result[e] = vs
        return result

    def enumerate_support(self, expand=True):
        """
        Enumerates over all spanning trees on ``self.num_vertices`` vertices.

        Note this is only defined for small num_vertices; for large
        num_vertices this raises ``NotImplementedError``.
        """
        V = self.num_vertices
        E = V - 1
        trees = _get_spanning_trees(V)
        result = torch.empty((len(trees), E, 2), dtype=torch.long)
        for i, tree in enumerate(trees):
            for e, (v1, v2) in enumerate(tree):
                result[i, e, 0] = v1
                result[i, e, 1] = v2
        return result


def find_complete_edge(v1, v2):
    """
    Find the edge index k of an unsorted pair of vertices (v1, v2).
    """
    if v2 < v1:
        v1, v2 = v2, v1
    return v1 + v2 * (v2 - 1) // 2


def make_complete_graph(num_vertices):
    """
    Constructs a complete graph.

    The pairing function is: ``k = v1 + v2 * (v2 - 1) // 2``

    :param int num_vertices: Number of vertices.
    :returns: A tuple with elements:
        V: Number of vertices.
        K: Number of edges.
        grid: a 3 x K grid of (edge, vertex, vertex) triples.
    """
    if num_vertices < 2:
        raise ValueError('PyTorch cannot handle zero-sized multidimensional tensors')
    V = num_vertices
    K = V * (V - 1) // 2
    grid = torch.zeros((3, K), dtype=torch.long)
    k = 0
    for v2 in range(V):
        for v1 in range(v2):
            grid[0, k] = k
            grid[1, k] = v1
            grid[2, k] = v2
            k += 1
    return grid


def remove_edge(grid, e2k, neighbors, components, e):
    """
    Remove an edge from a spanning tree.
    """
    k = e2k[e]
    v1 = grid[1, k].item()
    v2 = grid[2, k].item()
    neighbors[v1].remove(v2)
    neighbors[v2].remove(v1)
    pending = {v1}
    while pending:
        v1 = pending.pop()
        components[v1] = 1
        for v2 in neighbors[v1]:
            if not components[v2]:
                pending.add(v2)
    return k


def add_edge(grid, e2k, neighbors, components, e, k):
    """
    Add an edge connecting two components to create a spanning tree.
    """
    e2k[e] = k
    v1 = grid[1, k].item()
    v2 = grid[2, k].item()
    neighbors[v1].add(v2)
    neighbors[v2].add(v1)
    components[:] = 0


def find_valid_edges(components, valid_edges):
    """
    Find all edges between two components in a complete undirected graph.

    :param components: A [V]-shaped array of boolean component ids. This
        assumes there are exactly two nonemtpy components.
    :param valid_edges: An uninitialized array where output is written. On
        return, the subarray valid_edges[:end] will contain edge ids k for all
        valid edges.
    :returns: The number of valid edges found.
    """
    k = 0
    end = 0
    for v2, c2 in enumerate(components):
        for v1 in range(v2):
            if c2 ^ components[v1]:
                valid_edges[end] = k
                end += 1
            k += 1
    return end


def sample_tree(grid, edge_logits, edges):
    """
    Sample a random spanning tree of a dense weighted graph using MCMC.

    This uses Gibbs sampling on edges. Consider E undirected edges that can
    move around a graph of ``V=1+E`` vertices. The edges are constrained so
    that no two edges can span the same pair of vertices and so that the edges
    must form a spanning tree. To Gibbs sample, chose one of the E edges at
    random and move it anywhere else in the graph. After we remove the edge,
    notice that the graph is split into two connected components. The
    constraints imply that the edge must be replaced so as to connect the two
    components.  Hence to Gibbs sample, we collect all such bridging
    (vertex,vertex) pairs and sample from them in proportion to
    ``exp(edge_logits)``.

    :param grid: A 3 x K array as returned by :func:`make_complete_graph`.
    :param edge_logits: A length-K array of nonnormalized log probabilities.
    :param edges: A list of E initial edges in the form of (vertex,vertex) pairs.
    :returns: A list of ``(vertex, vertex)`` pairs.
    """
    if len(edges) <= 1:
        return edges
    E = len(edges)
    V = E + 1
    K = V * (V - 1) // 2
    e2k = torch.zeros(E, dtype=torch.long)
    neighbors = {v: set() for v in range(V)}
    components = torch.zeros(V, dtype=torch.uint8)
    for e in range(E):
        v1, v2 = edges[e]
        e2k[e] = find_complete_edge(v1, v2)
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    valid_edges = torch.empty(K, dtype=torch.long)

    for e in torch.randperm(E):  # Sequential scanning doesn't seem to work.
        e = e.item()
        k1 = remove_edge(grid, e2k, neighbors, components, e)
        num_valid_edges = find_valid_edges(components, valid_edges)
        valid_logits = edge_logits[valid_edges[:num_valid_edges]]
        valid_probs = torch.exp(valid_logits - valid_logits.max())
        total_prob = valid_probs.sum()
        if total_prob > 0:
            k2 = valid_edges[torch.multinomial(valid_probs, 1)[0]]
        else:
            k2 = k1
        add_edge(grid, e2k, neighbors, components, e, k2)

    edges = sorted((grid[1, k].item(), grid[2, k].item()) for k in e2k)
    assert len(edges) == E
    return edges


# FIXME This is probably an incorrect sampler.
@torch.no_grad()
def sample_tree_2(grid, edge_logits):
    """
    Sample a random spanning tree of a dense weighted graph.

    :param grid: A 3 x K array as returned by :func:`make_complete_graph`.
    :param edge_logits: A length-K array of nonnormalized log probabilities.
    :param edges: A list of E initial edges in the form of (vertex,vertex) pairs.
    :returns: A list of ``(vertex, vertex)`` pairs.
    """
    K = len(edge_logits)
    V = int(round(0.5 + (0.25 + 2 * K)**0.5))
    assert K == V * (V - 1) // 2
    E = V - 1
    components = edge_logits.new_zeros(V, dtype=torch.uint8)
    ks = []

    # Sample the first edge at random.
    probs = (edge_logits - edge_logits.max()).exp()
    k = torch.multinomial(probs, 1)[0]
    components[grid[1:, k]] = 1
    ks.append(k)

    # Sample edges connecting the cumulative tree to a new leaf.
    for e in range(1, E):
        c1, c2 = components[grid[1:]]
        mask = (c1 != c2)
        valid_logits = edge_logits[mask]
        probs = (valid_logits - valid_logits.max()).exp()
        k = mask.nonzero()[torch.multinomial(probs, 1)[0]]
        components[grid[1:, k]] = 1
        ks.append(k)

    edges = tuple((grid[1, k].item(), grid[2, k].item()) for k in sorted(ks))
    assert len(edges) == E
    return edges


_cpp_module = None
_cpp_source = """
#include <cmath>

at::Tensor make_complete_graph(long num_vertices) {
  const long V = num_vertices;
  const long K = V * (V - 1) / 2;
  auto grid = torch::empty({3, K}, at::kLong);
  int k = 0;
  for (int v2 = 0; v2 != V; ++v2) {
    for (int v1 = 0; v1 != v2; ++v1) {
      grid[0][k] = k;
      grid[1][k] = v1;
      grid[2][k] = v2;
      k += 1;
    }
  }
  return grid;
}

at::Tensor sample_tree(at::Tensor edge_logits) {
  torch::NoGradGuard no_grad;

  const long K = edge_logits.size(0);
  const long V = static_cast<long>(0.5 + std::sqrt(0.25 + 2 * K));
  const long E = V - 1;
  auto grid = make_complete_graph(V);
  auto components = torch::zeros({V}, at::kByte);
  auto ks = torch::empty({E}, at::kLong);

  // Sample the first edge at random.
  auto probs = (edge_logits - edge_logits.max()).exp();
  auto k = probs.multinomial(1)[0];
  components[grid[1][k]] = 1;
  components[grid[2][k]] = 1;
  ks[0] = k;

  // Sample edges connecting the cumulative tree to a new leaf.
  for (int e = 1; e != E; ++e) {
    auto c1 = components.index_select(0, grid[1]);
    auto c2 = components.index_select(0, grid[2]);
    auto mask = c1.__xor__(c2);
    auto valid_logits = edge_logits.masked_select(mask);
    auto probs = (valid_logits - valid_logits.max()).exp();
    auto k = mask.nonzero().view(-1)[probs.multinomial(1)[0]];
    components[grid[1][k]] = 1;
    components[grid[2][k]] = 1;
    ks[e] = k;
  }

  ks.sort();
  auto edges = torch::empty({E, 2}, at::kLong);
  for (int e = 0; e != E; ++e) {
    edges[e][0] = grid[1][ks[e]];
    edges[e][1] = grid[2][ks[e]];
  }
  return edges;
}
"""


def _get_cpp_module():
    """
    JIT compiles the cpp_spanning_tree module.
    """
    global _cpp_module
    if _cpp_module is None:
        import warnings
        from torch.utils.cpp_extension import load_inline
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            _cpp_module = load_inline(name="cpp_spanning_tree",
                                      cpp_sources=[_cpp_source],
                                      functions=["sample_tree"],
                                      extra_cflags=['-O2'],
                                      verbose=True)
    return _cpp_module


def sample_tree_3(edge_logits):
    return _get_cpp_module().sample_tree(edge_logits)


# See https://oeis.org/A000272
NUM_SPANNING_TREES = [
    1, 1, 1, 3, 16, 125, 1296, 16807, 262144, 4782969, 100000000, 2357947691,
    61917364224, 1792160394037, 56693912375296, 1946195068359375,
    72057594037927936, 2862423051509815793, 121439531096594251776,
    5480386857784802185939,
]

# These topologically distinct sets of trees generate sets of all trees
# under permutation of vertices. See https://oeis.org/A000055
_TREE_GENERATORS = [
    [[]],
    [[]],
    [[(0, 1)]],
    [[(0, 1), (0, 2)]],
    [
        [(0, 1), (0, 2), (0, 3)],
        [(0, 1), (1, 2), (2, 3)],
    ],
    [
        [(0, 1), (0, 2), (0, 3), (0, 4)],
        [(0, 1), (0, 2), (0, 3), (1, 4)],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
    ],
    [
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)],
        [(0, 1), (0, 2), (0, 3), (1, 4), (4, 5)],
        [(0, 1), (0, 2), (0, 3), (2, 4), (3, 5)],
        [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    ],
]


def _permute_tree(perm, tree):
    return tuple(sorted(tuple(sorted([perm[u], perm[v]])) for (u, v) in tree))


def _close_under_permutations(V, tree_generators):
    vertices = list(range(V))
    trees = []
    for tree in tree_generators:
        trees.extend(set(_permute_tree(perm, tree)
                         for perm in itertools.permutations(vertices)))
    trees.sort()
    return trees


def _get_spanning_trees(V):
    """
    Compute the set of spanning trees on V vertices.
    """
    if V >= len(_TREE_GENERATORS):
        raise NotImplementedError
    all_trees = _close_under_permutations(V, _TREE_GENERATORS[V])
    assert len(all_trees) == NUM_SPANNING_TREES[V]
    return all_trees
