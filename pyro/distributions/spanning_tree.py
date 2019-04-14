from __future__ import absolute_import, division, print_function

import itertools

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch_distribution import TorchDistribution


class SpanningTree(TorchDistribution):
    """
    Distribution over spanning trees on a fixed number ``V`` of vertices.

    :meth:`log_prob` is implemented using Kirchoff's theorem and a
    Cholesky-based computation of log determinant.

    :meth:`sample` is implemented using MCMC run for a small number of steps
    after being initialized by a cheap approximate sampler. This sampler is
    approximate and cubic time. This is much faster than the classic
    Aldous-Broder sampler [1,2] for graphs with large mixing time. Recent
    research [3,4] proposes samplers that run in sub-matrix-multiply time but
    are more complex to implement.

    :meth:`enumerate_support` is implemented for trees with up to 6 vertices.

    **References**

    [1] `Generating random spanning trees`
        Andrei Broder (1989)
    [2] `The Random Walk Construction of Uniform Spanning Trees and Uniform Labelled Trees`,
        David J. Aldous (1990)
    [3] `Sampling Random Spanning Trees Faster than Matrix Multiplication`,
        David Durfee, Rasmus Kyng, John Peebles, Anup B. Rao, Sushant Sachdeva
        (2017) https://arxiv.org/abs/1611.07451
    [4] `An almost-linear time algorithm for uniform random spanning tree generation`,
        Aaron Schild (2017) https://arxiv.org/abs/1711.06455

    :param torch.Tensor edge_logits: A tensor of length ``V*(V-1)//2``
        containing logits (aka negative energies) of all edges in the complete
        graph.  Edges in the complete graph are ordered:
        ``(0,1), (0,2), (1,2), (0,3), (1,3), (2,3), (0,4), (1,4), (1,5), ...``
    :param dict sampler_options: An optional dict of sampler options including:
        ``mcmc_steps`` defaulting to a single MCMC step (which is pretty good);
        ``initial_edges`` defaulting to a cheap approximate sample;
        ``backend`` one of "python" or "cpp", defaulting to "python".
    """
    arg_constraints = {'edge_logits': constraints.real}
    support = constraints.positive_integer
    has_enumerate_support = True

    def __init__(self, edge_logits, sampler_options=None, validate_args=None):
        if edge_logits.is_cuda:
            raise NotImplementedError("SpanningTree does not support cuda tensors")
        K = len(edge_logits)
        V = int(round(0.5 + (0.25 + 2 * K)**0.5))
        assert K == V * (V - 1) // 2
        E = V - 1
        event_shape = (E, 2)
        batch_shape = ()
        self.edge_logits = edge_logits
        super(SpanningTree, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        if self._validate_args:
            if edge_logits.shape != (K,):
                raise ValueError("Expected edge_logits of shape ({},), but got shape {}"
                                 .format(K, edge_logits.shape))
        self.num_vertices = V
        self.sampler_options = {} if sampler_options is None else sampler_options

    @lazy_property
    def log_partition_function(self):
        # By Kirchoff's matrix-tree theorem, the partition function is the
        # determinant of a truncated version of the graph Laplacian matrix. We
        # use a Cholesky decomposition to compute the log determinant.
        # See https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem
        V = self.num_vertices
        grid = make_complete_graph(V)
        shift = self.edge_logits.max()
        edge_probs = (self.edge_logits - shift).exp()
        adjacency = edge_probs.new_zeros(V, V)
        adjacency[grid[0], grid[1]] = edge_probs
        adjacency[grid[1], grid[0]] = edge_probs
        laplacian = adjacency.sum(-1).diag() - adjacency
        truncated = laplacian[:-1, :-1]
        log_det = torch.cholesky(truncated).diag().log().sum() * 2
        return log_det + shift * (V - 1)

    def log_prob(self, edges):
        if self._validate_args:
            if edges.dim() < 2 or edges.shape[-2:] != self.event_shape:
                raise ValueError("Invalid edges shape: {}".format(edges.shape))
        v1 = edges[..., 0]
        v2 = edges[..., 1]
        if self._validate_args:
            assert (v1 < v2).all()
        k = v1 + v2 * (v2 - 1) // 2
        return self.edge_logits[k].sum(-1) - self.log_partition_function

    def sample(self, sample_shape=torch.Size()):
        if sample_shape:
            raise NotImplementedError("SpanningTree does not support batching")
        edges = sample_tree(self.edge_logits, **self.sampler_options)
        assert edges.dim() >= 2 and edges.shape[-2:] == self.event_shape
        return edges

    def enumerate_support(self, expand=True):
        trees = enumerate_spanning_trees(self.num_vertices)
        return torch.tensor(trees, dtype=torch.long)


################################################################################
# Sampler implementation.
################################################################################

_cpp_module = None


def _get_cpp_module():
    """
    JIT compiles the cpp_spanning_tree module.
    """
    global _cpp_module
    if _cpp_module is None:
        import os
        import warnings
        from torch.utils.cpp_extension import load
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spanning_tree.cpp")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            _cpp_module = load(name="cpp_spanning_tree",
                               sources=[path],
                               extra_cflags=['-O2'],
                               verbose=True)
    return _cpp_module


def make_complete_graph(num_vertices, backend="python"):
    """
    Constructs a complete graph.

    The pairing function is: ``k = v1 + v2 * (v2 - 1) // 2``

    :param int num_vertices: Number of vertices.
    :returns: a 2 x K grid of (vertex, vertex) pairs.
    """
    if backend == "python":
        return _make_complete_graph(num_vertices)
    elif backend == "cpp":
        return _get_cpp_module().make_complete_graph(num_vertices)
    else:
        raise ValueError("unknown backend: {}".format(repr(backend)))


def _make_complete_graph(num_vertices):
    if num_vertices < 2:
        raise ValueError('PyTorch cannot handle zero-sized multidimensional tensors')
    V = num_vertices
    K = V * (V - 1) // 2
    v1 = torch.arange(V)
    v2 = torch.arange(V).unsqueeze(-1)
    v1, v2 = torch.broadcast_tensors(v1, v2)
    v1 = v1.contiguous().view(-1)
    v2 = v2.contiguous().view(-1)
    mask = (v1 < v2)
    grid = torch.stack((v1[mask], v2[mask]))
    assert grid.shape == (2, K)
    return grid


def _remove_edge(grid, e2k, neighbors, components, e):
    """
    Remove an edge from a spanning tree.
    """
    k = e2k[e]
    v1 = grid[0, k].item()
    v2 = grid[1, k].item()
    neighbors[v1].remove(v2)
    neighbors[v2].remove(v1)
    components[v1] = 1
    pending = [v1]
    while pending:
        v1 = pending.pop()
        for v2 in neighbors[v1]:
            if not components[v2]:
                components[v2] = 1
                pending.append(v2)
    return k


def _add_edge(grid, e2k, neighbors, components, e, k):
    """
    Add an edge connecting two components to create a spanning tree.
    """
    e2k[e] = k
    v1 = grid[0, k].item()
    v2 = grid[1, k].item()
    neighbors[v1].add(v2)
    neighbors[v2].add(v1)
    components.fill_(0)


def _find_valid_edges(components, valid_edges):
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


@torch.no_grad()
def _sample_tree_mcmc(edge_logits, edges):
    if len(edges) <= 1:
        return edges

    E = len(edges)
    V = E + 1
    K = V * (V - 1) // 2
    grid = make_complete_graph(V)
    e2k = torch.empty(E, dtype=torch.long)
    neighbors = {v: set() for v in range(V)}
    components = torch.zeros(V, dtype=torch.uint8)
    for e in range(E):
        v1, v2 = map(int, edges[e])
        assert v1 < v2
        e2k[e] = v1 + v2 * (v2 - 1) // 2
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    valid_edges_buffer = torch.empty(K, dtype=torch.long)

    for e in range(E):
        k = _remove_edge(grid, e2k, neighbors, components, e)
        num_valid_edges = _find_valid_edges(components, valid_edges_buffer)
        valid_edges = valid_edges_buffer[:num_valid_edges]
        valid_logits = edge_logits[valid_edges]
        valid_probs = (valid_logits - valid_logits.max()).exp()
        total_prob = valid_probs.sum()
        if total_prob > 0:
            sample = torch.multinomial(valid_probs, 1)[0]
            k = valid_edges[sample]
        _add_edge(grid, e2k, neighbors, components, e, k)

    e2k.sort()
    edges = edge_logits.new_empty((E, 2), dtype=torch.long)
    edges[:, 0] = grid[0, e2k]
    edges[:, 1] = grid[1, e2k]
    return edges


def sample_tree_mcmc(edge_logits, edges, backend="python"):
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

    :param torch.Tensor edge_logits: A length-K array of nonnormalized log
        probabilities.
    :param torch.Tensor edges: An E x 2 tensor of initial edges in the form
        of (vertex,vertex) pairs. Each edge should be sorted and the entire
        tensor should be lexicographically sorted.
    :returns: An E x 2 tensor of edges in the form of (vertex,vertex) pairs.
        Each edge should be sorted and the entire tensor should be
        lexicographically sorted.
    :rtype: torch.Tensor
    """
    if backend == "python":
        return _sample_tree_mcmc(edge_logits, edges)
    elif backend == "cpp":
        return _get_cpp_module().sample_tree_mcmc(edge_logits, edges)
    else:
        raise ValueError("unknown backend: {}".format(repr(backend)))


@torch.no_grad()
def _sample_tree_approx(edge_logits):
    K = len(edge_logits)
    V = int(round(0.5 + (0.25 + 2 * K)**0.5))
    assert K == V * (V - 1) // 2
    E = V - 1
    grid = make_complete_graph(V)
    components = edge_logits.new_zeros(V, dtype=torch.uint8)
    e2k = edge_logits.new_empty((E,), dtype=torch.long)

    # Sample the first edge at random.
    probs = (edge_logits - edge_logits.max()).exp()
    k = torch.multinomial(probs, 1)[0]
    components[grid[:, k]] = 1
    e2k[0] = k

    # Sample edges connecting the cumulative tree to a new leaf.
    for e in range(1, E):
        c1, c2 = components[grid]
        mask = (c1 != c2)
        valid_logits = edge_logits[mask]
        probs = (valid_logits - valid_logits.max()).exp()
        k = mask.nonzero()[torch.multinomial(probs, 1)[0]]
        components[grid[:, k]] = 1
        e2k[e] = k

    e2k.sort()
    edges = edge_logits.new_empty((E, 2), dtype=torch.long)
    edges[:, 0] = grid[0, e2k]
    edges[:, 1] = grid[1, e2k]
    return edges


def sample_tree_approx(edge_logits, backend="python"):
    """
    Approximately sample a random spanning tree of a dense weighted graph.

    This is mainly useful for initializing an MCMC sampler.

    :param torch.Tensor edge_logits: A length-K array of nonnormalized log
        probabilities.
    :returns: An E x 2 tensor of edges in the form of (vertex,vertex) pairs.
        Each edge should be sorted and the entire tensor should be
        lexicographically sorted.
    :rtype: torch.Tensor
    """
    if backend == "python":
        return _sample_tree_approx(edge_logits)
    elif backend == "cpp":
        return _get_cpp_module().sample_tree_approx(edge_logits)
    else:
        raise ValueError("unknown backend: {}".format(repr(backend)))


def sample_tree(edge_logits, init_edges=None, mcmc_steps=1, backend="python"):
    edges = init_edges
    if edges is None:
        edges = sample_tree_approx(backend=backend)
    for step in range(mcmc_steps):
        edges = sample_tree_mcmc(edge_logits, edges, backend=backend)
    return edges


################################################################################
# Enumeration implementation.
################################################################################

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


def enumerate_spanning_trees(V):
    """
    Compute the set of spanning trees on V vertices.
    """
    if V >= len(_TREE_GENERATORS):
        raise NotImplementedError
    all_trees = _close_under_permutations(V, _TREE_GENERATORS[V])
    assert len(all_trees) == NUM_SPANNING_TREES[V]
    return all_trees
