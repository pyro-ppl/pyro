from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from six.moves import range

from pyro.distributions.torch_distribution import TorchDistribution


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


class SpanningTree(TorchDistribution):
    """
    Distribution over spanning trees on a fixed set of vertices.
    """
    arg_constraings = {'edge_logits': constraints.real}
    support = constraints.positive_integer

    def __init__(self, edge_logits, initial_edges=None, mcmc_steps=1, validate_args=None):
        K = len(edge_logits)
        V = int(round(0.5 + (0.25 + 2 * K)**0.5))
        assert K == V * (V - 1) // 2
        E = V - 1
        event_shape = (E, 2)
        batch_shape = ()
        super(SpanningTree, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        if initial_edges is None:
            initial_edges = torch.stack((torch.arange(0, V - 1), torch.arange(1, V)), dim=-1)
        if self._validate_args:
            if edge_logits.shape != (K, 2):
                raise ValueError("Expected edge_logits of shape ({},2), but got shape {}"
                                 .format(K, edge_logits.shape))
            if initial_edges.shape != (E, 2):
                raise ValueError("Expected initial_edges of shape ({},2), but got shape {}"
                                 .format(K, edge_logits.shape))
        self.edge_logits = edge_logits
        self.initial_edges = initial_edges
        self.mcmc_steps = mcmc_steps
        self.complete_graph = make_complete_graph(V)

    def log_prob(self, edges):
        return self.edge_logits.new_tensor(0.)

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
