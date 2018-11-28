from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from contextlib2 import ExitStack

import torch

import pyro
import pyro.distributions as dist
import pyro.ops.packed as packed
import pyro.poutine as poutine

from pyro.ops.contract import contract_tensor_tree
from pyro.ops.einsum.adjoint import require_backward
from pyro.ops.rings import SampleRing


class DebugRing(SampleRing):

    def sumproduct(self, terms, dims):
        v = super(DebugRing, self).sumproduct(terms, dims)
        print("DEBUG sumproduct: dims {}, terms {}, result {}".format(dims, terms, v))
        return v

    def product(self, term, ordinal):
        v = super(DebugRing, self).product(term, ordinal)
        print("DEBUG product: ordinal {}, term {}, result {}".format(ordinal, term, v))
        return v

    def broadcast(self, term, ordinal):
        v = super(DebugRing, self).broadcast(term, ordinal)
        print("DEBUG broadcast: ordinal {}, term {}, result {}".format(ordinal, term, v))
        return v

    def inv(self, term):
        v = super(DebugRing, self).inv(term)
        print("DEBUG inv: term {}, result {}".format(term, v))
        return v


class CollapseEnumMessenger(poutine.enumerate_messenger.EnumerateMessenger):

    def _pyro_sample(self, msg):
        collapse = msg["infer"].get("collapse")
        if collapse:
            msg["infer"]["enumerate"] = "parallel"

        super(CollapseEnumMessenger, self)._pyro_sample(msg)

        # if collapse:
        #     msg["is_observed"] = True


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
        dim_to_frame = {}
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
                        dim_to_frame[frame_dim] = frame
                        sum_dims.remove(frame_dim)

                # Note we mark all sites with require_backward to get correct ordinals and slice non-enumerated samples
                if not node["is_observed"]:
                    queries.append(log_prob)
                    require_backward(log_prob)

        ring = DebugRing()
        contract_tensor_tree(log_probs, sum_dims, ring=ring)
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
                new_node = {}
                log_prob = node["packed"]["log_prob"]
                ordinal = query_ordinal[log_prob]
                new_node["cond_indep_stack"] = tuple(
                    f for f in node["cond_indep_stack"]
                    if not f.vectorized or frame_to_dim[f] in ordinal)

                new_node["infer"] = node["infer"].copy()

                # TODO move this into a custom SampleRing Leaf implementation
                sample = log_prob._pyro_backward_result
                # sample_dim = log_prob._pyro_dims[-1]
                new_value = node["value"]
                for index, dim in zip(sample, sample._pyro_sample_dims):
                    if dim in new_value._pyro_dims:
                        index._pyro_dims = sample._pyro_dims[1:]
                        new_value = packed.gather(new_value, index, dim)
                new_node["value"] = new_value

                collapsed_trace.add_node(node["name"], **new_node)

        # Add new observe sites induced by moralization
        i = 0
        for ordinal, terms in log_probs.items():
            for term in terms:
                with ExitStack() as stack:
                    for dim in ordinal:
                        frame = dim_to_frame[dim]
                        stack.enter_context(pyro.plate(frame.name, frame.size, dim=frame.dim))
                    pyro.sample("aux_{}".format(i),
                                dist.Bernoulli(probs=torch.exp(-term / 2.)),
                                obs=torch.tensor(1.))
                i += 1

        # Replay model correctly against collapsed_trace (get correct cond_indep_stack)
        collapsed_sites = set(node["name"] for node in enum_trace.nodes.values()
                              if node["type"] == "sample" and node["infer"].get("collapse"))

        with poutine.block(hide_fn=lambda msg: msg["name"] in collapsed_sites):
            with CollapseReplayMessenger(trace=collapsed_trace):
                return model(*args, **kwargs)

    return _collapsed_model
