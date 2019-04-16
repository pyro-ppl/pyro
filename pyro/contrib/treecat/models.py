from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.distributions.spanning_tree import make_complete_graph
from pyro.infer.discrete import infer_discrete


class EdgeGuide(object):
    """
    Conjugate guide for latent z distribution.
    """
    def __init__(self, capacity, edges):
        E = len(edges)
        V = E + 1
        K = V * (V - 1) // 2
        M = capacity
        self.capacity = capacity
        self.edges = edges
        self._grid = make_complete_graph(V)

        self._count_prior = 0.5 * M
        self._count_stats = 0.
        self._vertex_prior = 0.5
        self._vertex_stats = torch.zeros((V, M), dtype=torch.float)
        self._edge_prior = 0.5 / M
        self._edge_stats = torch.zeros((E, M * M), dtype=torch.float)
        self._complete_stats = torch.zeros((K, M * M), dtype=torch.float)

    @torch.no_grad()
    def get_posterior(self):
        count = self._count_prior + self.count_stats
        vertex_probs = (self._vertex_prior + self._vertex_stats) / count
        edge_probs = (self._edge_prior + self._edge_stats) / count
        return vertex_probs, edge_probs

    @torch.no_grad()
    def update_stats(self, data, num_rows, z):
        E = len(self.edges)
        V = E - 1
        K = V * (V - 1) // 2
        M = self.capacity
        batch_size = z.shape(-1)
        assert z.dim() == 2

        decay = 1. - batch_size / num_rows
        self._count_stats *= decay
        self._vertex_stats *= decay
        self._edge_stats *= decay
        self._complete_stats *= decay

        self._count_stats += batch_size
        # TODO Find a vectorized version of this.
        for i in range(batch_size):
            z_i = z[:, i]
            self._vertex_stats[torch.arange(V), z_i] += 1
            zz_i = M * z_i[self.edges[:, 0]] + z_i[self.edges[:, 1]]
            self._edge_stats[torch.arange(E), zz_i] += 1
            zz_i = M * z_i[self.grid[0]] + z_i[self.grid[1]]
            self._complete_stats[torch.arange(K), zz_i] += 1


class TreeCat(object):
    """
    The TreeCat model of sparse heterogeneous tabular data.

    This is intended to be trained with
    :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO` and served with
    :func:`~pyro.infer.discrete.infer_discrete`, and requires
    ``max_plate_nesting >= 1``.

    :param list features: a ``V``-lenth list of
        :class:`~pyro.contrib.tabular.features.Feature` objects defining a
        feature model for each column. Feature models can be repeated,
        indicating that two columns share a common feature model (with shared
        learned parameters).
    :param int capacity: cardinality of latent categorical variables
    :param tuple edges: an ``(V-1) x 2`` nested tuple representing the tree
        structure. Each of the ``E = (V-1)`` edges is a tuple ``v1,v2`` of
        vertices.
    """
    def __init__(self, features, capacity, edges):
        assert capacity > 1
        assert len(edges) == len(features) - 1
        assert all(len(edge) == 2 for edge in edges)
        self.features = features
        self.capacity = capacity
        self.edges = edges

        self._feature_guide = AutoDelta(poutine.block(
            self.model, hide_fn=lambda msg: msg["name"].startswith("treecat_")))
        self._edge_guide = EdgeGuide(num_vertices=len(features), capacity=capacity)
        self._saved_z = None

    def model(self, data, num_rows=None, batch_size=None, impute=False):
        """
        :param list data: batch of heterogeneous column-oriented data.  Each
            column should be either a torch.Tensor (if oberved) or None (if
            unobserved).
        :param bool impute: Whether to impute missing features. This should be
            set to False during training and True when making predictions.
        :returns: a copy of the input data, optionally with missing columns
            stochastically imputed according the the joint posterior.
        :rtype: list
        """
        assert len(data) == len(self.features)
        assert any(column is not None for column in data)
        if num_rows is None:
            for column in data:
                if column is not None:
                    num_rows = column.size(0)
                    break
        V = len(self.features)
        E = len(self.edges)
        M = self.capacity

        # TODO fix AutoDelta to support sequential pyro.plate.
        # vertices_plate = pyro.plate("vertices_range", V)
        # edges_plate = pyro.plate("edges_range", E)
        vertices_plate = range(V)
        edges_plate = range(E)
        components_plate = pyro.plate("components_plate", M)

        # Sample a mixture model for each feature.
        mixtures = [None] * V
        prev_features = {}  # allows feature models to be shared by multiple columns
        for v in vertices_plate:
            feature = self.features[v]
            prev_v = prev_features.setdefault(id(feature), v)
            if prev_v != v:
                mixtures[v] = mixtures[prev_v]
            else:
                shared = feature.sample_shared()
                with components_plate:
                    mixtures[v] = feature.sample_group(shared)

        # Sample latent vertex- and edge- distributions from a Dirichlet prior.
        with pyro.plate("vertices_plate", V):
            vertex_probs = pyro.sample("treecat_vertex_probs",
                                       dist.Dirichlet(self._vertex_prior))
        with pyro.plate("edges_plate", E):
            edge_probs = pyro.sample("treecat_edge_probs",
                                     dist.Dirichlet(self._edge_prior))

        # Sample data-local variables.
        subsample = None if batch_size is None else [None] * batch_size
        with pyro.plate("data", num_rows, subsample=subsample):

            # Sample discrete latent state from an undirected tree structure.
            z = [None] * V
            for v in vertices_plate:
                z[v] = pyro.sample("treecat_z_{}".format(v),
                                   dist.Categorical(vertex_probs[v]),
                                   infer={"enumerate": "parallel"})
            for e in edges_plate:
                v1, v2 = map(int, self.edges[e])
                probs = (edge_probs[e].reshape(M, M) /
                         vertex_probs[v1].unsqueeze(-1) /
                         vertex_probs[v2]).reshape(M * M)
                pyro.sample("treecat_z_{}_{}".format(v1, v2),
                            dist.Categorical(probs),
                            obs=M * z[v1] + z[v2])
            self._saved_z = z

            # Sample observed features conditioned on latent classes.
            x = [None] * V
            for v in vertices_plate:
                if data[v] is None and not impute:
                    continue
                component_dist = self.features[v].value_dist(mixtures[v], component=z[v])
                x[v] = pyro.sample("treecat_x_{}".format(v), component_dist,
                                   obs=data[v])

        return x

    def guide(self, data, num_rows=None, batch_size=None, impute=False):
        """
        A :class:`~pyro.contrib.autoguide.AutoDelta` guide for MAP inference of
        continuous parameters.
        """
        V = len(self.features)
        E = V - 1

        self._feature_guide(data, num_rows=num_rows, impute=impute)

        vertex_probs, edge_probs = self._edge_guide.get_posterior()
        with pyro.plate("vertices_plate", V):
            pyro.sample("treecat_vertex_probs", dist.Delta(vertex_probs, event_dim=1))
        with pyro.plate("edges_plate", E):
            pyro.sample("treecat_edge_probs", dist.Delta(edge_probs, event_dim=1))

        pyro.plate("data", num_rows, dim=-1)

    def _update_stats(self, data, num_rows, batch_size):
        guide_trace = poutine.trace(self.guide).get_trace(data)
        model = poutine.replay(self.model, guide_trace=guide_trace)
        model = infer_discrete(model, first_available_dim=-2)
        model(data, num_rows, batch_size, impute=False)
        z = torch.stack(self._saved_z)
        self._edge_guide.update_stats(data, num_rows, batch_size, z)
