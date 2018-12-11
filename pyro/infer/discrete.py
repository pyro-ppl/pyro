from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

import pyro.ops.packed as packed
from pyro import poutine
from pyro.ops.contract import contract_tensor_tree
from pyro.ops.einsum.adjoint import require_backward
from pyro.ops.rings import MapRing, SampleRing
from pyro.poutine.enumerate_messenger import EnumerateMessenger
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.util import prune_subsample_sites
from pyro.util import jit_iter

_RINGS = {0: MapRing, 1: SampleRing}


def _make_ring(temperature):
    try:
        return _RINGS[temperature]()
    except KeyError:
        raise ValueError("temperature must be 0 (map) or 1 (sample) for now")


class SamplePosteriorMessenger(ReplayMessenger):
    # This acts like ReplayMessenger but additionally replays cond_indep_stack.

    def _pyro_sample(self, msg):
        if msg["infer"].get("enumerate") == "parallel":
            super(SamplePosteriorMessenger, self)._pyro_sample(msg)
        if msg["name"] in self.trace:
            msg["cond_indep_stack"] = self.trace.nodes[msg["name"]]["cond_indep_stack"]


def _sample_posterior(model, first_available_dim, temperature, *args, **kwargs):
    # For internal use by infer_discrete.

    # Create an enumerated trace.
    with poutine.block(), EnumerateMessenger(first_available_dim):
        enum_trace = poutine.trace(model).get_trace(*args, **kwargs)
    enum_trace = prune_subsample_sites(enum_trace)
    enum_trace.compute_log_prob()
    enum_trace.pack_tensors()
    plate_to_symbol = enum_trace.plate_to_symbol

    # Collect a set of query sample sites to which the backward algorithm will propagate.
    log_probs = OrderedDict()
    sum_dims = set()
    queries = []
    for node in enum_trace.nodes.values():
        if node["type"] == "sample":
            ordinal = frozenset(plate_to_symbol[f.name]
                                for f in node["cond_indep_stack"] if f.vectorized)
            log_prob = node["packed"]["log_prob"]
            log_probs.setdefault(ordinal, []).append(log_prob)
            sum_dims.update(log_prob._pyro_dims)
            for frame in node["cond_indep_stack"]:
                if frame.vectorized:
                    sum_dims.remove(plate_to_symbol[frame.name])
            # Note we mark all sample sites with require_backward to gather
            # enumerated sites and adjust cond_indep_stack of all sample sites.
            if not node["is_observed"]:
                queries.append(log_prob)
                require_backward(log_prob)

    # Run forward-backward algorithm, collecting the ordinal of each connected component.
    ring = _make_ring(temperature)
    log_probs = contract_tensor_tree(log_probs, sum_dims, ring=ring)  # run forward algorithm
    query_to_ordinal = {}
    pending = object()  # a constant value for pending queries
    for query in queries:
        query._pyro_backward_result = pending
    for ordinal, terms in log_probs.items():
        for term in terms:
            if hasattr(term, "_pyro_backward"):
                term._pyro_backward()  # run backward algorithm
        # Note: this is quadratic in number of ordinals
        for query in queries:
            if query not in query_to_ordinal and query._pyro_backward_result is not pending:
                query_to_ordinal[query] = ordinal

    # Construct a collapsed trace by gathering and adjusting cond_indep_stack.
    collapsed_trace = poutine.Trace()
    for node in enum_trace.nodes.values():
        if node["type"] == "sample" and not node["is_observed"]:
            # TODO move this into a Leaf implementation somehow
            new_node = {
                "type": "sample",
                "name": node["name"],
                "is_observed": False,
                "infer": node["infer"].copy(),
                "cond_indep_stack": node["cond_indep_stack"],
                "value": node["value"],
            }
            log_prob = node["packed"]["log_prob"]
            if hasattr(log_prob, "_pyro_backward_result"):
                # Adjust the cond_indep_stack.
                ordinal = query_to_ordinal[log_prob]
                new_node["cond_indep_stack"] = tuple(
                    f for f in node["cond_indep_stack"]
                    if not f.vectorized or plate_to_symbol[f.name] in ordinal)

                # Gather if node depended on an enumerated value.
                sample = log_prob._pyro_backward_result
                if sample is not None:
                    new_value = packed.pack(node["value"], node["infer"]["_dim_to_symbol"])
                    for index, dim in zip(jit_iter(sample), sample._pyro_sample_dims):
                        if dim in new_value._pyro_dims:
                            index._pyro_dims = sample._pyro_dims[1:]
                            new_value = packed.gather(new_value, index, dim)
                    new_node["value"] = packed.unpack(new_value, enum_trace.symbol_to_dim)

            collapsed_trace.add_node(node["name"], **new_node)

    # Replay the model against the collapsed trace.
    with SamplePosteriorMessenger(trace=collapsed_trace):
        return model(*args, **kwargs)


def infer_discrete(fn=None, first_available_dim=None, temperature=1):
    """
    A poutine that samples discrete sites marked with
    ``site["infer"]["enumerate"] = "parallel"`` from the posterior,
    conditioned on observations.

    Example::

        @infer_discrete(first_available_dim=-1, temperature=0)
        @config_enumerate
        def viterbi_decoder(data, hidden_dim=10):
            transition = 0.3 / hidden_dim + 0.7 * torch.eye(hidden_dim)
            means = torch.arange(float(hidden_dim))
            states = [0]
            for t in pyro.markov(range(len(data))):
                states.append(pyro.sample("states_{}".format(t),
                                          dist.Categorical(transition[states[-1]])))
                pyro.sample("obs_{}".format(t),
                            dist.Normal(means[states[-1]], 1.),
                            obs=data[t])
            return states  # returns maximum likelihood states

    .. warning: The ``log_prob``s of the inferred model's trace are not
        meaningful, and may be changed future release.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This should be a negative integer.
    :param int temperature: Either 1 (sample via forward-filter backward-sample)
        or 0 (optimize via Viterbi-like MAP inference). Defaults to 1 (sample).
    """
    assert first_available_dim < 0, first_available_dim
    if fn is None:  # support use as a decorator
        return functools.partial(infer_discrete,
                                 first_available_dim=first_available_dim,
                                 temperature=temperature)
    return functools.partial(_sample_posterior, fn, first_available_dim, temperature)
