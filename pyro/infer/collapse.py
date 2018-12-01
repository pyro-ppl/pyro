from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pyro.ops.packed as packed
import pyro.poutine as poutine

from pyro.ops.contract import contract_tensor_tree
from pyro.ops.einsum.adjoint import require_backward
from pyro.ops.rings import SampleRing


class CollapseEnumMessenger(poutine.enumerate_messenger.EnumerateMessenger):

    def _pyro_sample(self, msg):
        collapse = msg["infer"].get("collapse")
        if collapse:
            msg["infer"]["enumerate"] = "parallel"

        super(CollapseEnumMessenger, self)._pyro_sample(msg)


class CollapseReplayMessenger(poutine.replay_messenger.ReplayMessenger):

    def _pyro_sample(self, msg):
        if msg["infer"].get("collapse"):
            super(CollapseReplayMessenger, self)._pyro_sample(msg)
            msg["stop"] = True
        if msg["name"] in self.trace:
            msg["cond_indep_stack"] = self.trace.nodes[msg["name"]]["cond_indep_stack"]


def collapse(model, first_available_dim):
    """
    Use `ubersum` to collapse sample sites marked with `site["infer"]["collapse"] = True`

    .. warning:: Cannot be wrapped with :func:~`pyro.poutine.replay`
    """

    def _collapsed_model(*args, **kwargs):
        with poutine.block():
            enum_trace = poutine.trace(
                CollapseEnumMessenger(first_available_dim)(model)
            ).get_trace(*args, **kwargs)

        enum_trace = poutine.util.prune_subsample_sites(enum_trace)
        enum_trace.compute_log_prob()
        enum_trace.pack_tensors()

        log_probs = OrderedDict()
        frame_to_dim = {}
        sum_dims = set()
        queries = []
        for node in enum_trace.nodes.values():
            if node["type"] == "sample":

                log_prob = node["packed"]["log_prob"]
                ordinal = frozenset(log_prob._pyro_dims[f.dim] for f in node["cond_indep_stack"] if f.vectorized)
                log_probs.setdefault(ordinal, []).append(log_prob)
                sum_dims.update(set(log_prob._pyro_dims))

                for frame in node["cond_indep_stack"]:
                    if frame.vectorized:
                        frame_dim = log_prob._pyro_dims[frame.dim]
                        frame_to_dim[frame] = frame_dim
                        sum_dims.remove(frame_dim)

                # Note we mark all sites with require_backward to get correct ordinals and slice non-enumerated samples
                if not node["is_observed"]:
                    queries.append(log_prob)
                    require_backward(log_prob)

        ring = SampleRing()
        log_probs = contract_tensor_tree(log_probs, sum_dims, ring=ring)
        query_ordinal = {}
        for ordinal, terms in log_probs.items():
            for term in terms:
                term._pyro_backward()
            # Note: makes collapse quadratic in number of ordinals
            for query in queries:
                if query not in query_ordinal and query._pyro_backward_result is not None:
                    query_ordinal[query] = ordinal

        collapsed_trace = poutine.Trace()
        for node in enum_trace.nodes.values():
            if node["type"] == "sample" and not node["is_observed"]:
                # TODO move this into a Leaf implementation somehow
                new_node = {"type": "sample", "name": node["name"], "is_observed": False}
                log_prob = node["packed"]["log_prob"]
                ordinal = query_ordinal[log_prob]
                new_node["cond_indep_stack"] = tuple(
                    f for f in node["cond_indep_stack"]
                    if not f.vectorized or frame_to_dim[f] in ordinal)

                new_node["infer"] = node["infer"].copy()

                # TODO move this into a custom SampleRing Leaf implementation
                sample = log_prob._pyro_backward_result
                new_value = packed.pack(node["value"], node["infer"]["_dim_to_symbol"])
                for index, dim in zip(sample, sample._pyro_sample_dims):
                    if dim in new_value._pyro_dims:
                        index._pyro_dims = sample._pyro_dims[1:]
                        new_value = packed.gather(new_value, index, dim)

                new_node["value"] = packed.unpack(new_value, enum_trace.symbol_to_dim)

                collapsed_trace.add_node(node["name"], **new_node)

        # Replay model correctly against collapsed_trace (get correct cond_indep_stack)
        collapsed_sites = set(node["name"] for node in enum_trace.nodes.values()
                              if node["type"] == "sample" and node["infer"].get("collapse"))

        with poutine.block(hide_fn=lambda msg: msg["name"] in collapsed_sites):
            with CollapseReplayMessenger(trace=collapsed_trace):
                return model(*args, **kwargs)

    return _collapsed_model
