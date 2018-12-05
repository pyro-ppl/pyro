from __future__ import absolute_import, division, print_function

import functools
from collections import OrderedDict

import pyro.ops.packed as packed
import pyro.poutine as poutine
from pyro.ops.contract import contract_tensor_tree
from pyro.ops.einsum.adjoint import require_backward
from pyro.ops.rings import MapRing, SampleRing

_RINGS = {0: MapRing, 1: SampleRing}


def _make_ring(temperature):
    try:
        return _RINGS[temperature]()
    except KeyError:
        raise ValueError("temperature must be 0 (map) or 1 (sample) for now")


class CollapseEnumMessenger(poutine.enumerate_messenger.EnumerateMessenger):

    def _pyro_sample(self, msg):
        collapse = msg["infer"].get("collapse")
        if collapse:
            msg["infer"]["enumerate"] = "parallel"

        super(CollapseEnumMessenger, self)._pyro_sample(msg)


class SamplePosteriorMessenger(poutine.replay_messenger.ReplayMessenger):

    def _pyro_sample(self, msg):
        if msg["infer"].get("collapse"):
            super(SamplePosteriorMessenger, self)._pyro_sample(msg)
        if msg["name"] in self.trace:
            msg["cond_indep_stack"] = self.trace.nodes[msg["name"]]["cond_indep_stack"]


def _sample_posterior(model, first_available_dim, temperature, *args, **kwargs):
    # Create an enumerated trace.
    with poutine.block():
        enum_trace = poutine.trace(
            CollapseEnumMessenger(first_available_dim)(model)
        ).get_trace(*args, **kwargs)
    enum_trace = poutine.util.prune_subsample_sites(enum_trace)
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

            # Note we mark all sites with require_backward to get correct
            # ordinals and slice non-enumerated samples.
            if not node["is_observed"]:
                queries.append(log_prob)
                require_backward(log_prob)

    # Run forward-backward algorithm, collecting the ordinal of each connected component.
    ring = _make_ring(temperature)
    log_probs = contract_tensor_tree(log_probs, sum_dims, ring=ring)
    query_to_ordinal = {}
    pending = object()  # a constant value for pending queries
    for query in queries:
        query._pyro_backward_result = pending
    for ordinal, terms in log_probs.items():
        for term in terms:
            if hasattr(term, "_pyro_backward"):
                term._pyro_backward()
        # Note: this is quadratic in number of ordinals
        for query in queries:
            if query not in query_to_ordinal and query._pyro_backward_result is not pending:
                query_to_ordinal[query] = ordinal

    # Construct a collapsed trace by slicing and adjusting cond_indep_stack.
    collapsed_trace = poutine.Trace()
    for node in enum_trace.nodes.values():
        if node["type"] == "sample" and not node["is_observed"]:
            # TODO move this into a Leaf implementation somehow
            new_node = {"type": "sample", "name": node["name"], "is_observed": False}
            log_prob = node["packed"]["log_prob"]
            new_node["infer"] = node["infer"].copy()

            if hasattr(log_prob, "_pyro_backward"):
                ordinal = query_to_ordinal[log_prob]
                new_node["cond_indep_stack"] = tuple(
                    f for f in node["cond_indep_stack"]
                    if not f.vectorized or plate_to_symbol[f.name] in ordinal)

                # TODO move this into a custom SampleRing Leaf implementation
                sample = log_prob._pyro_backward_result
                if sample is None:
                    # node did not depend on an enumerated variable, so no sampling necessary
                    new_node["value"] = node["value"]
                else:
                    new_value = packed.pack(node["value"], node["infer"]["_dim_to_symbol"])
                    for index, dim in zip(sample, sample._pyro_sample_dims):
                        if dim in new_value._pyro_dims:
                            index._pyro_dims = sample._pyro_dims[1:]
                            new_value = packed.gather(new_value, index, dim)
                    new_node["value"] = packed.unpack(new_value, enum_trace.symbol_to_dim)
            else:
                new_node["cond_indep_stack"] = node["cond_indep_stack"]
                new_node["value"] = node["value"]

            collapsed_trace.add_node(node["name"], **new_node)

    # Replay the model against the collapsed trace.
    with SamplePosteriorMessenger(trace=collapsed_trace):
        return model(*args, **kwargs)


def sample_posterior(model, first_available_dim, temperature=1):
    """
    A handler that samples sites marked with
    ``site["infer"]["collapse"] = True`` from the posterior.
    Such sites will appear with sampled values.

    .. warning:: Cannot be wrapped with :func:~`pyro.poutine.replay`

    :param model: a stochastic function (callable containing Pyro primitive calls)
    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This should be a negative integer.
    :param int temperature: Either 1 (sample via forward-filter backward-sample)
        or 0 (optimize via Viterbi-like MAP inference). Defaults to 1 (sample).
    """
    return functools.partial(_sample_posterior, model, first_available_dim, temperature)


def _is_collapsed(node):
    return node["type"] == "sample" and node["infer"].get("collapse")


def collapse(model, first_available_dim):
    """
    A handler to collapse sample sites marked with
    ``site["infer"]["collapse"] = True``.
    Collapsed sites will be blocked.

    .. warning:: Cannot be wrapped with :func:~`pyro.poutine.replay`

    :param model: a stochastic function (callable containing Pyro primitive calls)
    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This should be a negative integer.
    """
    collapsed_model = sample_posterior(model, first_available_dim, temperature=1)
    return poutine.block(collapsed_model, hide_fn=_is_collapsed)
