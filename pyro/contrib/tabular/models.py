from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta


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
        self._guide = AutoDelta(poutine.block(
            self.model, hide_fn=lambda msg: msg["name"].startswith("treecat_")))

    def model(self, data, impute=False):
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
        for column in data:
            if column is not None:
                batch_size = len(column)
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
            vertex_probs = pyro.sample("vertex_probs",
                                       dist.Dirichlet(0.5 * torch.ones(M)))
        with pyro.plate("edges_plate", E):
            edge_probs = pyro.sample("edge_probs",
                                     dist.Dirichlet(0.5 / M * torch.ones(M * M)))

        # Sample data-local variables.
        with pyro.plate("data", batch_size) as row_ids:

            # Sample discrete latent state from an undirected tree structure.
            z = [None] * V
            for v in vertices_plate:
                z[v] = pyro.sample("treecat_z_{}".format(v),
                                   dist.Categorical(vertex_probs[v]),
                                   infer={'enumerate': 'parallel'})
            for e in edges_plate:
                v1, v2 = self.edges[e]
                probs = (edge_probs[e].reshape(M, M) /
                         vertex_probs[v1].unsqueeze(-1) /
                         vertex_probs[v2]).reshape(M * M)
                pyro.sample("treecat_z_{}_{}".format(v1, v2),
                            dist.Categorical(probs),
                            obs=M * z[v1] + z[v2])

            # Sample observed features conditioned on latent classes.
            x = [None] * V
            for v in vertices_plate:
                column = data[v]
                if column is None and not impute:
                    continue
                component_dist = self.features[v].value_dist(mixtures[v], component=z[v])
                x[v] = pyro.sample("treecat_x_{}".format(v), component_dist,
                                   obs=None if column is None else column[row_ids])

        return x

    def guide(self, data, impute=False):
        """
        A :class:`~pyro.contrib.autoguide.AutoDelta` guide for MAP inference of
        continuous parameters plus exact inference of discrete latent
        variables.
        """
        return self._guide(data, impute=impute)
