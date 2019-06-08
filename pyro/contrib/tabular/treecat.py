from __future__ import absolute_import, division, print_function

import logging
import re
from collections import OrderedDict, deque

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.distributions.spanning_tree import make_complete_graph, sample_tree_mcmc
from pyro.infer import SVI
from pyro.infer.discrete import TraceEnumSample_ELBO, infer_discrete
from pyro.ops import packed
from pyro.ops.contract import contract_to_tensor
from pyro.ops.indexing import Vindex


class TreeCat(object):
    """
    The TreeCat model of sparse heterogeneous tabular data.

    **Serialization:** :class:`TreeCat` models distribute state between a
    :class:`TreeCat` object and the Pyro param store. Thus to save and load a
    model you'll need to combine Pyro's
    :meth:`~pyro.params.param_store.ParamStoreDict.save` /
    :meth:`~pyro.params.param_store.ParamStoreDict.load` with
    :py:func:`pickle.dump` / :py:func:`pickle.load` . For example::

        model = TreeCat(...)
        # ...train model...

        # Save model.
        pyro.get_param_store().save("model.pyro")
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Load model.
        pyro.get_param_store().load("model.pyro")
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

    :param list features: A ``V``-length list of
        :class:`~pyro.contrib.tabular.features.Feature` objects defining a
        feature model for each column. Feature models can be repeated,
        indicating that two columns share a common feature model (with shared
        learned parameters). Features should reside on the same device as data.
    :param int capacity: Cardinality of latent categorical variables.
    :param torch.LongTensor edges: A ``(V-1, 2)`` shaped tensor representing
        the tree structure. Each of the ``E = (V-1)`` edges is a row ``v1,v2``
        of vertices. Edges must reside on the CPU.
    :param float annealing_rate: The exponential growth rate limit with which
        sufficient statistics approach the full dataset early in training.
        Should be positive.
    """
    def __init__(self, features, capacity=8, edges=None, annealing_rate=0.01):
        V = len(features)
        E = V - 1
        M = capacity
        if edges is None:
            edges = torch.stack([torch.arange(E, device="cpu"),
                                 torch.arange(1, 1 + E, device="cpu")], dim=-1)
        assert capacity > 1
        assert isinstance(edges, torch.LongTensor)  # Note edges must live on CPU.
        assert edges.shape == (E, 2)
        self.features = features
        self.capacity = capacity

        self._feature_guide = AutoDelta(poutine.block(
            self.model, hide_fn=lambda msg: msg["name"].startswith("treecat_")))
        self._edge_guide = _EdgeGuide(capacity=capacity, edges=edges, annealing_rate=annealing_rate)
        self._vertex_prior = torch.full((M,), 1.)
        self._edge_prior = torch.full((M * M,), 1. / M)
        self._saved_z = None

        self.edges = edges

    # These custom __getstate__ and __setstate__ methods are used by the pickle
    # module for model serialization.
    def __getstate__(self):
        edges = self.edges.tolist()
        init_args = (self.features, self.capacity, edges)
        return {"init_args": init_args, "edge_guide": self._edge_guide}

    def __setstate__(self, state):
        features, capacity, edges = state["init_args"]
        edges = torch.tensor(edges, device="cpu")
        self.__init__(features, capacity, edges)
        self._edge_guide = state["edge_guide"]
        self._edge_guide.edges = self.edges

    @property
    def edges(self):
        """
        A ``(V-1, 2)`` shaped tensor representing the tree structure.
        You can examine the tree using :func:`print_tree` ::

            print(print_tree(model.edges, model.features))
        """
        return self._edges

    @edges.setter
    def edges(self, edges):
        self._edges = edges
        self._edge_guide.edges = edges

        # Construct a directed tree data structure used by ._propagate().
        # The root has no statistical meaning; we choose a root vertex based on
        # computational concerns only, maximizing opportunity for parallelism.
        self._root = find_center_of_tree(edges)
        self._neighbors = [set() for _ in self.features]
        self._edge_index = {}
        for e, (v1, v2) in enumerate(edges.numpy()):
            self._neighbors[v1].add(v2)
            self._neighbors[v2].add(v1)
            self._edge_index[v1, v2] = e
            self._edge_index[v2, v1] = e

    def _validate_data_mask(self, data, mask):
        assert len(data) == len(self.features)
        assert all(isinstance(col, torch.Tensor) for col in data)
        if mask is None:
            mask = [True] * len(self.features)
        assert len(mask) == len(self.features)
        assert all(isinstance(col, (torch.Tensor, bool)) for col in mask)
        return data, mask

    def model(self, data, mask=None, num_rows=None, impute=False):
        """
        The generative model of tabular data.

        :param list data: A minibatch of column-oriented data. Each column
            should be a :class:`torch.Tensor` .
        :param list mask: A minibatch of column masks. Each column may be
            ``True`` if fully observed, ``False`` if fully unobserved, or a
            :class:`torch.ByteTensor` if partially observed.
        :param int num_rows: The total number of rows in the dataset.
            This is needed only when subsampling data.
        :param bool impute: Whether to impute missing features. This should be
            set to False during training and True when making predictions.
        :returns: a copy of the input data, optionally with missing columns
            stochastically imputed according the joint posterior.
        :rtype: list
        """
        data, mask = self._validate_data_mask(data, mask)
        device = data[0].device
        batch_size = len(data[0])
        if num_rows is None:
            num_rows = batch_size
        V = len(self.features)
        E = len(self.edges)
        M = self.capacity

        # Sample a mixture model for each feature.
        mixtures = [None] * V
        components_plate = pyro.plate("components_plate", M, dim=-1)
        for v, feature in enumerate(self.features):
            shared = feature.sample_shared()
            with components_plate:
                mixtures[v] = feature.sample_group(shared)

        # Sample latent vertex- and edge- distributions from a Dirichlet prior.
        with pyro.plate("vertices_plate", V, dim=-1):
            vertex_probs = pyro.sample("treecat_vertex_probs",
                                       dist.Dirichlet(self._vertex_prior.to(device)))
        with pyro.plate("edges_plate", E, dim=-1):
            edge_probs = pyro.sample("treecat_edge_probs",
                                     dist.Dirichlet(self._edge_prior.to(device)))
        if vertex_probs.dim() > 2:
            vertex_probs = vertex_probs.unsqueeze(-3)
            edge_probs = edge_probs.unsqueeze(-3)

        # Sample data-local variables.
        subsample = None if (batch_size == num_rows) else [None] * batch_size
        with pyro.plate("data", num_rows, subsample=subsample, dim=-1):

            # Recursively sample z and x in Markov contexts.
            z = [None] * V
            x = list(data)
            v = self._root
            self._propagate(data, mask, impute, mixtures, vertex_probs, edge_probs, z, x, v)

        self._saved_z = [z_v.cpu() for z_v in z]
        return x

    @poutine.markov
    def _propagate(self, data, mask, impute, mixtures, vertex_probs, edge_probs, z, x, v):
        # Determine the upstream parent v0 and all downstream children.
        v0 = None
        children = []
        for v2 in self._neighbors[v]:
            if z[v2] is None:
                children.append(v2)
            else:
                v0 = v2

        # Sample discrete latent state from an arbitrarily directed tree structure.
        M = self.capacity
        if v0 is None:
            # Sample root node unconditionally.
            probs = vertex_probs[..., v, :]
        else:
            # Sample node v conditioned on its parent v0.
            joint = edge_probs[..., self._edge_index[v, v0], :]
            joint = joint.reshape(joint.shape[:-1] + (M, M))
            if v0 > v:
                joint = joint.transpose(-1, -2)
            probs = Vindex(joint)[..., z[v0], :]
        z[v] = pyro.sample("treecat_z_{}".format(v),
                           # Avoid spurious validation errors of constraints.simplex.
                           dist.Categorical(probs, validate_args=(probs.size(-1) <= 8)),
                           infer={"enumerate": "parallel"})

        # Sample observed features conditioned on latent classes.
        x_dist = self.features[v].value_dist(mixtures[v], component=z[v])
        if mask[v] is True:  # All rows are observed.
            pyro.sample("treecat_x_obs_{}".format(v), x_dist, obs=data[v])
        elif mask[v] is False:  # No rows are observed.
            if impute:
                x[v] = pyro.sample("treecat_x_{}".format(v), x_dist)
        else:  # Some rows are observed.
            with poutine.mask(mask=mask[v]):
                pyro.sample("treecat_x_obs_{}".format(v), x_dist, obs=data[v])
            if impute:
                with poutine.mask(mask=~mask[v]):
                    x[v] = pyro.sample("treecat_x_{}".format(v), x_dist)
                # Interleave conditioned and sampled data.
                mask_v = (slice(None),) * (x[v].dim() - data[v].dim()) + (mask[v],)
                x[v][mask_v] = data[v][mask[v]]

        # Continue sampling downstream.
        for v2 in children:
            self._propagate(data, mask, impute, mixtures, vertex_probs, edge_probs, z, x, v2)

    def guide(self, data, mask=None, num_rows=None, impute=False):
        """
        A :class:`~pyro.contrib.autoguide.AutoDelta` guide for MAP inference of
        continuous parameters.

        :param list data: A minibatch of column-oriented data. Each column
            should be a :class:`torch.Tensor` .
        :param list mask: A minibatch of column masks. Each column may be
            ``True`` if fully observed, ``False`` if fully unobserved, or a
            :class:`torch.ByteTensor` if partially observed.
        :param int num_rows: The total number of rows in the dataset.
            This is needed only when subsampling data.
        :param bool impute: Whether to impute missing features. This should be
            set to False during training and True when making predictions.
        """
        data, mask = self._validate_data_mask(data, mask)
        device = data[0].device
        V = len(self.features)
        E = V - 1

        self._feature_guide(data, num_rows=num_rows, impute=impute)

        # This guide uses the posterior mean as a point estimate.
        vertex_probs, edge_probs = self._edge_guide.get_posterior()
        with pyro.plate("vertices_plate", V, dim=-1):
            pyro.sample("treecat_vertex_probs",
                        dist.Delta(vertex_probs.to(device), event_dim=1))
        with pyro.plate("edges_plate", E, dim=-1):
            pyro.sample("treecat_edge_probs",
                        dist.Delta(edge_probs.to(device), event_dim=1))

    def trainer(self, optim, backend="cpp"):
        """
        Creates a :class:`TreeCatTrainer` object for training.

        :param pyro.optim.optim.PyroOptim optim: A Pyro optimizer to learn
            parameters, e.g. :class:`~pyro.optim.pytorch_optimizers.Adam` .
        :param str backend: Either "python" or "cpp". Defaults to "cpp". The
            "cpp" backend is much faster for data with more than ~10 features.
        :rtype: TreeCatTrainer
        """
        return TreeCatTrainer(self, optim, backend)

    def sample(self, data, mask=None, num_samples=None):
        """
        Sample missing data conditioned on observed data.

        :param list data: A minibatch of column-oriented data. Each column
            should be a :class:`torch.Tensor` .
        :param list mask: A minibatch of column masks. Each column may be
            ``True`` if fully observed, ``False`` if fully unobserved, or a
            :class:`torch.ByteTensor` if partially observed.
        :param int num_samples: Optional number of samples to draw.
        """
        model = self.model
        guide = self.guide
        first_available_dim = -2

        # Optionally draw vectorized samples.
        if num_samples is not None:
            plate = pyro.plate("num_samples_vectorized", num_samples,
                               dim=first_available_dim)
            model = plate(model)
            guide = plate(guide)
            first_available_dim -= 1

        # Sample global parameters from the guide.
        guide_trace = poutine.trace(guide).get_trace(data, mask)

        # Sample local variables using variable elimination.
        return infer_discrete(
            poutine.replay(model, guide_trace),
            first_available_dim=first_available_dim)(data, mask, impute=True)

    def log_prob(self, data, mask=None):
        """
        Compute posterior predictive probabilities of partially observed rows.

        :param list data: A minibatch of column-oriented data. Each column
            should be a :class:`torch.Tensor` .
        :param list mask: A minibatch of column masks. Each column may be
            ``True`` if fully observed, ``False`` if fully unobserved, or a
            :class:`torch.ByteTensor` if partially observed.
        :return: A batch of posterior predictive log probabilities with one
            entry per row.
        :rtype: torch.Tensor
        """
        # Trace the guide and model.
        guide_trace = poutine.trace(self.guide).get_trace(data, mask)
        model_trace = poutine.trace(
            poutine.replay(self.model, guide_trace)).get_trace(data, mask)
        model_trace.compute_log_prob()
        model_trace.pack_tensors()

        # Perform variable elimination, preserving the data plate.
        ordinal = frozenset({model_trace.plate_to_symbol["data"]})
        log_factors = [packed.scale_and_mask(site["packed"]["unscaled_log_prob"],
                                             mask=site["packed"]["mask"])
                       for name, site in model_trace.nodes.items()
                       if re.match("treecat_[zx]_.*", name)]
        sum_dims = set.union(*(set(x._pyro_dims) for x in log_factors)) - ordinal
        tensor_tree = OrderedDict({ordinal: log_factors})
        log_prob = contract_to_tensor(tensor_tree, sum_dims, ordinal)
        assert log_prob.shape == (len(data[0]),)
        return log_prob


class TreeCatTrainer(object):
    """
    Maintains state to initialize and train a :class:`TreeCat` model.

    :param TreeCat model: A :class:`TreeCat` model to train.
    :param pyro.optim.optim.PyroOptim optim: A Pyro optimizer to learn feature
        parameters, e.g. :class:`~pyro.optim.pytorch_optimizers.Adam` .
    :param str backend: Either "python" or "cpp". Defaults to "cpp". The
        "cpp" backend is much faster for data with more than ~10 features.
    """
    def __init__(self, model, optim, backend="cpp"):
        assert isinstance(model, TreeCat)
        self.backend = backend
        self._elbo = TraceEnumSample_ELBO(max_plate_nesting=1)
        self._svi = SVI(model.model, model.guide, optim, self._elbo)
        self._model = model
        self._initialized = False

    def init(self, data, mask=None):
        """
        Initializes shared feature parameters given some or all data.

        :param list data: A minibatch of column-oriented data. Each column
            should be a :class:`torch.Tensor` .
        :param list mask: A minibatch of column masks. Each column may be
            ``True`` if fully observed, ``False`` if fully unobserved, or a
            :class:`torch.ByteTensor` if partially observed.
        """
        data, mask = self._model._validate_data_mask(data, mask)
        for feature, col_data, col_mask in zip(self._model.features, data, mask):
            if col_data is not None and col_mask is not False:
                if isinstance(col_mask, torch.Tensor):
                    col_data = col_data[col_mask]
                feature.init(col_data)
        self._initialized = True

    def step(self, data, mask=None, num_rows=None):
        """
        Runs one training step.

        :param list data: A minibatch of column-oriented data. Each column
            should be a :class:`torch.Tensor` .
        :param list mask: A minibatch of column masks. Each column may be
            ``True`` if fully observed, ``False`` if fully unobserved, or a
            :class:`torch.ByteTensor` if partially observed.
        :param int num_rows: The total number of rows in the dataset.
            This is needed only when subsampling data.
        """
        if not self._initialized:
            self.init(data, mask)

        # Perform a gradient optimizer step to learn parameters.
        loss = self._svi.step(data, mask, num_rows=num_rows)

        # Update sufficient statistics in the edge guide.
        self._elbo.sample_saved()
        z = torch.stack(self._model._saved_z)
        self._model._edge_guide.update(num_rows, z)

        # Perform an MCMC step to learn the model.
        model = self._model
        edge_logits = model._edge_guide.compute_edge_logits()
        model.edges = sample_tree_mcmc(edge_logits, model.edges, backend=self.backend)

        return loss


class _EdgeGuide(object):
    """
    Conjugate guide for latent categorical distribution parameters.

    .. note:: This is memory intensive and therefore resides on the CPU.

    :param int capacity: The cardinality of discrete latent variables.
    :param torch.LongTensor edges: A ``(V-1, 2)`` shaped tensor representing
        the tree structure. Each of the ``E = (V-1)`` edges is a row ``v1,v2``
        of vertices.
    :param float annealing_rate: The exponential growth rate limit with which
        sufficient statistics approach the full dataset early in training.
        Should be positive.
    """
    def __init__(self, capacity, edges, annealing_rate):
        assert 0 < annealing_rate
        E = len(edges)
        V = E + 1
        K = V * (V - 1) // 2
        M = capacity
        self.capacity = capacity
        self.edges = edges
        self.annealing_rate = annealing_rate
        self._grid = make_complete_graph(V)

        # Use a uniform prior on vertices, forcing a sparse prior on edges.
        self._vertex_prior = 1.  # A uniform Dirichlet of shape (M,).
        self._edge_prior = 1. / M  # A uniform Dirichlet of shape (M,M).

        # Initialize stats to a single pseudo-observation.
        self._count_stats = 1.
        self._vertex_stats = torch.full((V, M), 1. / M, device="cpu")
        self._complete_stats = torch.full((K, M * M), 1. / M ** 2, device="cpu")

    @torch.no_grad()
    def update(self, num_rows, z):
        """
        Updates count statistics given a minibatch of latent samples ``z``.

        :param int num_rows: Size of the complete dataset.
        :param torch.Tensor z: A minibatch of latent variables of size
            ``(V, batch_size)``.
        """
        assert z.dim() == 2
        M = self.capacity
        batch_size = z.size(-1)
        if num_rows is None:
            num_rows = batch_size

        # Early in learning, we limit stats accumulation to slow exponential
        # growth determined by annealing_rate. Later in learning we
        # exponentially smooth batches to approximate the entire dataset.
        assert batch_size <= num_rows
        assert self._count_stats > 0
        annealing = (1 + self.annealing_rate) / (1 + batch_size / self._count_stats)
        exponential_smoothing = 1 / (1 + batch_size / num_rows)
        decay = min(annealing, exponential_smoothing)

        # Accumulate statistics and decay.
        self._count_stats += batch_size
        self._count_stats *= decay
        one = self._vertex_stats.new_tensor(1.)
        self._vertex_stats.scatter_add_(-1, z, one.expand_as(z))
        self._vertex_stats *= decay
        zz = (M * z)[self._grid[0]] + z[self._grid[1]]
        self._complete_stats.scatter_add_(-1, zz, one.expand_as(zz))
        self._complete_stats *= decay

        # Log metrics to diagnose convergence issues.
        if logging.Logger(None).isEnabledFor(logging.DEBUG):
            logging.debug("count_stats = {:0.1f}, batch_size = {}, num_rows = {}".format(
                self._count_stats, batch_size, num_rows))

            # Compute empirical perplexity of latent variables.
            vertex_probs = self._vertex_stats + 1e-6
            vertex_probs /= vertex_probs.sum(-1, True)
            vertex_entropy = -(vertex_probs * vertex_probs.log()).sum(-1)
            perplexity = vertex_entropy.exp().sort(descending=True)[0]
            perplexity = ["{: >4.1f}".format(p) for p in perplexity]
            logging.debug(" ".join(["perplexity:"] + perplexity))

            # Compute empirical mutual information across edges.
            v1, v2 = self.edges.t()
            k = v1 + v2 * (v2 - 1) // 2
            edge_probs = self._complete_stats[k] + 1e-6
            edge_probs /= edge_probs.sum(-1, True)
            edge_entropy = -(edge_probs * edge_probs.log()).sum(-1)
            mutual_info = vertex_entropy[self.edges].sum(-1) - edge_entropy
            mutual_info = mutual_info.sort(descending=True)[0]
            mutual_info = ["{: >4.1f}".format(i) for i in mutual_info]
            logging.debug(" ".join(["mutual_info:"] + mutual_info))

    @torch.no_grad()
    def get_posterior(self):
        """
        Computes posterior mean under a Dirichlet prior.

        :returns: a pair ``vetex_probs,edge_probs`` with the posterior mean
            probabilities of each of the ``V`` latent variables and pairwise
            probabilities of each of the ``K=V*(V-1)/2`` pairs of latent
            variables.
        :rtype: tuple
        """
        v1, v2 = self.edges.t()
        k = v1 + v2 * (v2 - 1) // 2
        edge_stats = self._complete_stats[k]

        vertex_probs = self._vertex_prior + self._vertex_stats
        vertex_probs /= vertex_probs.sum(-1, True)
        edge_probs = self._edge_prior + edge_stats
        edge_probs /= edge_probs.sum(-1, True)
        return vertex_probs, edge_probs

    @torch.no_grad()
    def compute_edge_logits(self):
        """
        Computes a non-normalized log likelihoods of each of the
        ``K=V*(V-1)/2`` edges in the complete graph. This can be used to learn
        tree structure distributed according to the
        :class:`~pyro.distributions.SpanningTree` distribution.

        :returns: a ``(K,)``-shaped tensor of edges logits.
        :rtype: torch.Tensor
        """
        E = len(self.edges)
        V = E + 1
        K = V * (V - 1) // 2
        vertex_logits = _dirmul_log_prob(self._vertex_prior, self._vertex_stats)
        edge_logits = _dirmul_log_prob(self._edge_prior, self._complete_stats)
        edge_logits -= vertex_logits[self._grid[0]]
        edge_logits -= vertex_logits[self._grid[1]]
        assert edge_logits.shape == (K,)
        return edge_logits


def _dirmul_log_prob(alpha, counts):
    """
    Computes non-normalized log probability of a Dirichlet-multinomial
    distribution in a numerically stable way. Equivalent to::

        (alpha + counts).lgamma().sum(-1) - (1 + counts).lgamma().sum(-1)
    """
    assert isinstance(alpha, float)
    shape = counts.shape
    temp = (counts.unsqueeze(-1) + counts.new_tensor([alpha, 1])).lgamma_()
    temp = temp.reshape(-1, 2).mv(temp.new_tensor([1., -1.])).reshape(shape)
    return temp.sum(-1)


def find_center_of_tree(edges):
    """
    Finds a maximally central vertex in a tree.

    :param torch.LongTensor edges: A ``(V-1, 2)`` shaped tensor representing
        the tree structure. Each of the ``E = (V-1)`` edges is a row ``v1,v2``
        of vertices.
    :returns: Vertex id of a maximally central vertex.
    :rtype: int
    """
    V = len(edges) + 1
    neighbors = [set() for _ in range(V)]
    for v1, v2 in edges.numpy():
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    queue = deque(v for v in range(V) if len(neighbors[v]) <= 1)
    while queue:
        v = queue.popleft()
        for v2 in sorted(neighbors[v]):
            neighbors[v2].remove(v)
            if len(neighbors[v2]) == 1:
                queue.append(v2)
    return v


def print_tree(edges, feature_names, root=None):
    """
    Returns a text representation of the feature tree.

    :param torch.LongTensor edges: A ``(V-1, 2)`` shaped tensor representing
        the tree structure. Each of the ``E = (V-1)`` edges is a row ``v1,v2``
        of vertices.
    :param list feature_names: A list of feature names.
    :param str root: The name of the root feature (optional).
    :returns: A text representation of the tree with one feature per line.
    :rtype: str
    """
    assert len(feature_names) == 1 + len(edges)
    if root is None:
        root = feature_names[find_center_of_tree(edges)]
    assert root in feature_names
    neighbors = [set() for _ in feature_names]
    for v1, v2 in edges.numpy():
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    stack = [feature_names.index(root)]
    seen = set(stack)
    lines = []
    while stack:
        backtrack = True
        for neighbor in sorted(neighbors[stack[-1]], reverse=True):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
                backtrack = False
                break
        if backtrack:
            name = feature_names[stack.pop()]
            lines.append((len(stack), name))
    lines.reverse()
    return "\n".join(["{}{}".format("  " * i, n) for i, n in lines])
