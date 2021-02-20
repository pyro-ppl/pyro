# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch_distribution import TorchDistribution


class SpanningTree(TorchDistribution):
    """
    Distribution over spanning trees on a fixed number ``V`` of vertices.

    A tree is represented as :class:`torch.LongTensor` ``edges`` of shape
    ``(V-1,2)`` satisfying the following properties:

    1. The edges constitute a tree, i.e. are connected and cycle free.
    2. Each edge ``(v1,v2) = edges[e]`` is sorted, i.e. ``v1 < v2``.
    3. The entire tensor is sorted in colexicographic order.

    Use :func:`validate_edges` to verify `edges` are correctly formed.

    The ``edge_logits`` tensor has one entry for each of the ``V*(V-1)//2``
    edges in the complete graph on ``V`` vertices, where edges are each sorted
    and the edge order is colexicographic::

        (0,1), (0,2), (1,2), (0,3), (1,3), (2,3), (0,4), (1,4), (2,4), ...

    This ordering corresponds to the size-independent pairing function::

        k = v1 + v2 * (v2 - 1) // 2

    where ``k`` is the rank of the edge ``(v1,v2)`` in the complete graph.
    To convert a matrix of edge logits to the linear representation used here::

        assert my_matrix.shape == (V, V)
        i, j = make_complete_graph(V)
        edge_logits = my_matrix[i, j]

    :param torch.Tensor edge_logits: A tensor of length ``V*(V-1)//2``
        containing logits (aka negative energies) of all edges in the complete
        graph on ``V`` vertices. See above comment for edge ordering.
    :param dict sampler_options: An optional dict of sampler options including:
        ``mcmc_steps`` defaulting to a single MCMC step (which is pretty good);
        ``initial_edges`` defaulting to a cheap approximate sample;
        ``backend`` one of "python" or "cpp", defaulting to "python".
    """
    arg_constraints = {'edge_logits': constraints.real}
    support = constraints.nonnegative_integer
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
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        if self._validate_args:
            if edge_logits.shape != (K,):
                raise ValueError("Expected edge_logits of shape ({},), but got shape {}"
                                 .format(K, edge_logits.shape))
        self.num_vertices = V
        self.sampler_options = {} if sampler_options is None else sampler_options

    def validate_edges(self, edges):
        """
        Validates a batch of ``edges`` tensors, as returned by :meth:`sample` or
        :meth:`enumerate_support` or as input to :meth:`log_prob()`.

        :param torch.LongTensor edges: A batch of edges.
        :raises: ValueError
        :returns: None
        """
        if edges.shape[-2:] != self.event_shape:
            raise ValueError("Invalid edges shape: {}".format(edges.shape))

        # Verify canonical ordering.
        if not ((0 <= edges) & (edges < self.num_vertices)).all():
            raise ValueError("Invalid vertex ids:\n{}".format(edges))
        if not (edges[..., 0] < edges[..., 1]).all():
            raise ValueError("Vertices are not sorted in each edge:\n{}".format(edges))
        if not ((edges[..., :-1, 1] < edges[..., 1:, 1]) |
                ((edges[..., :-1, 1] == edges[..., 1:, 1]) &
                 (edges[..., :-1, 0] < edges[..., 1:, 0]))).all():
            raise ValueError("Edges are not sorted colexicographically:\n{}".format(edges))

        # Verify tree property, i.e. connectivity.
        V = self.num_vertices
        for i in itertools.product(*map(range, edges.shape[:-2])):
            edges_i = edges[i]
            connected = torch.eye(V, dtype=torch.float)
            connected[edges_i[:, 0], edges_i[:, 1]] = 1
            connected[edges_i[:, 1], edges_i[:, 0]] = 1
            for i in range(int(math.ceil(V ** 0.5))):
                connected = connected.mm(connected).clamp_(max=1)
            if not connected.min() > 0:
                raise ValueError("Edges do not constitute a tree:\n{}".format(edges_i))

    @lazy_property
    def log_partition_function(self):
        # By Kirchoff's matrix-tree theorem, the partition function is the
        # determinant of a truncated version of the graph Laplacian matrix. We
        # use a Cholesky decomposition to compute the log determinant.
        # See https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem
        V = self.num_vertices
        v1, v2 = make_complete_graph(V).unbind(0)
        logits = self.edge_logits.new_full((V, V), -math.inf)
        logits[v1, v2] = self.edge_logits
        logits[v2, v1] = self.edge_logits
        log_diag = logits.logsumexp(-1)
        # Numerically stabilize so that laplacian has 1's on the diagonal.
        shift = 0.5 * log_diag
        laplacian = torch.eye(V) - (logits - shift - shift[:, None]).exp()
        truncated = laplacian[:-1, :-1]
        try:
            import gpytorch
            log_det = gpytorch.lazy.NonLazyTensor(truncated).logdet()
        except ImportError:
            log_det = torch.cholesky(truncated).diag().log().sum() * 2
        return log_det + log_diag[:-1].sum()

    def log_prob(self, edges):
        if self._validate_args:
            self.validate_edges(edges)
        v1 = edges[..., 0]
        v2 = edges[..., 1]
        k = v1 + v2 * (v2 - 1) // 2
        return self.edge_logits[k].sum(-1) - self.log_partition_function

    def sample(self, sample_shape=torch.Size()):
        """
        This sampler is implemented using MCMC run for a small number of steps
        after being initialized by a cheap approximate sampler. This sampler is
        approximate and cubic time. This is faster than the classic
        Aldous-Broder sampler [1,2], especially for graphs with large mixing
        time. Recent research [3,4] proposes samplers that run in
        sub-matrix-multiply time but are more complex to implement.

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
        """
        if sample_shape:
            raise NotImplementedError("SpanningTree does not support batching")
        edges = sample_tree(self.edge_logits, **self.sampler_options)
        assert edges.dim() >= 2 and edges.shape[-2:] == self.event_shape
        return edges

    def enumerate_support(self, expand=True):
        """
        This is implemented for trees with up to 6 vertices (and 5 edges).
        """
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
        from torch.utils.cpp_extension import load
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spanning_tree.cpp")
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


def _remove_edge(grid, edge_ids, neighbors, components, e):
    """
    Remove an edge from a spanning tree.
    """
    k = edge_ids[e]
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


def _add_edge(grid, edge_ids, neighbors, components, e, k):
    """
    Add an edge connecting two components to create a spanning tree.
    """
    edge_ids[e] = k
    v1 = grid[0, k].item()
    v2 = grid[1, k].item()
    neighbors[v1].add(v2)
    neighbors[v2].add(v1)
    components.fill_(0)


def _find_valid_edges(components, valid_edge_ids):
    """
    Find all edges between two components in a complete undirected graph.

    :param components: A [V]-shaped array of boolean component ids. This
        assumes there are exactly two nonemtpy components.
    :param valid_edge_ids: An uninitialized array where output is written. On
        return, the subarray valid_edge_ids[:end] will contain edge ids k for all
        valid edges.
    :returns: The number of valid edges found.
    """
    k = 0
    end = 0
    for v2, c2 in enumerate(components):
        for v1 in range(v2):
            if c2 ^ components[v1]:
                valid_edge_ids[end] = k
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

    # Each of E edges in the tree is stored as an id k in [0, K) indexing into
    # the complete graph. The id of an edge (v1,v2) is k = v1+v2*(v2-1)/2.
    edge_ids = torch.empty(E, dtype=torch.long)
    # This maps each vertex to the set of its neighboring vertices.
    neighbors = {v: set() for v in range(V)}
    # This maps each vertex to its connected component id (0 or 1).
    components = torch.zeros(V, dtype=torch.bool)
    for e in range(E):
        v1, v2 = map(int, edges[e])
        assert v1 < v2
        edge_ids[e] = v1 + v2 * (v2 - 1) // 2
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    # This stores ids of edges that are valid candidates for Gibbs moves.
    valid_edges_buffer = torch.empty(K, dtype=torch.long)

    # Cycle through all edges in a random order.
    for e in torch.randperm(E):
        e = int(e)

        # Perform a single-site Gibbs update by moving this edge elsewhere.
        k = _remove_edge(grid, edge_ids, neighbors, components, e)
        num_valid_edges = _find_valid_edges(components, valid_edges_buffer)
        valid_edge_ids = valid_edges_buffer[:num_valid_edges]
        valid_logits = edge_logits[valid_edge_ids]
        valid_probs = (valid_logits - valid_logits.max()).exp()
        total_prob = valid_probs.sum()
        if total_prob > 0:
            sample = torch.multinomial(valid_probs, 1)[0]
            k = valid_edge_ids[sample]
        _add_edge(grid, edge_ids, neighbors, components, e, k)

    # Convert edge ids to a canonical list of pairs.
    edge_ids = edge_ids.sort()[0]
    edges = torch.empty((E, 2), dtype=torch.long)
    edges[:, 0] = grid[0, edge_ids]
    edges[:, 1] = grid[1, edge_ids]
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

    # Each of E edges in the tree is stored as an id k in [0, K) indexing into
    # the complete graph. The id of an edge (v1,v2) is k = v1+v2*(v2-1)/2.
    edge_ids = torch.empty((E,), dtype=torch.long)
    # This maps each vertex to whether it is a member of the cumulative tree.
    components = torch.zeros(V, dtype=torch.bool)

    # Sample the first edge at random.
    probs = (edge_logits - edge_logits.max()).exp()
    k = torch.multinomial(probs, 1)[0]
    components[grid[:, k]] = 1
    edge_ids[0] = k

    # Sample edges connecting the cumulative tree to a new leaf.
    for e in range(1, E):
        c1, c2 = components[grid]
        mask = (c1 != c2)
        valid_logits = edge_logits[mask]
        probs = (valid_logits - valid_logits.max()).exp()
        k = mask.nonzero(as_tuple=False)[torch.multinomial(probs, 1)[0]]
        components[grid[:, k]] = 1
        edge_ids[e] = k

    # Convert edge ids to a canonical list of pairs.
    edge_ids = edge_ids.sort()[0]
    edges = torch.empty((E, 2), dtype=torch.long)
    edges[:, 0] = grid[0, edge_ids]
    edges[:, 1] = grid[1, edge_ids]
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
        edges = sample_tree_approx(edge_logits, backend=backend)
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
    edges = [tuple(sorted([perm[u], perm[v]])) for (u, v) in tree]
    edges.sort(key=lambda uv: (uv[1], uv[0]))
    return tuple(edges)


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
        raise NotImplementedError(
            "enumerate_spanning_trees() is implemented only for trees with up to {} vertices"
            .format(len(_TREE_GENERATORS) - 1))
    all_trees = _close_under_permutations(V, _TREE_GENERATORS[V])
    assert len(all_trees) == NUM_SPANNING_TREES[V]
    return all_trees
