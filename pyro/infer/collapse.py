from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import itertools

import torch
from opt_einsum import shared_intermediates
from torch.distributions.utils import broadcast_all

import pyro
import pyro.distributions as dist
from pyro.distributions.util import logsumexp

import pyro.poutine as poutine
from pyro.ops.contract import contract_to_tensor

from .traceenum_elbo import _make_dist


class CollapseEnumMessenger(poutine.enumerate_messenger.EnumerateMessenger):

    def _pyro_sample(self, msg):
        collapse = msg["infer"].get("collapse")
        if collapse:
            msg["infer"]["enumerate"] = "parallel"

        super(CollapseEnumMessenger, self)._pyro_sample(msg)

        if collapse:
            msg["is_observed"] = True


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

        prune_subsample_sites(enum_trace)
        enum_trace.compute_log_prob()
        enum_trace.pack_tensors()

        log_probs = OrderedDict()
        dim_to_frame = {}
        frame_to_dim = {}
        sum_dims = set()
        queries = []
        for node in enum_trace.values():
            if node["type"] == "sample":

                log_prob = node["packed"]["log_prob"]
                ordinal = frozenset(log_prob._pyro_dims[f.dim] for f in node["cond_indep_stack"] if f.vectorized)
                log_probs.setdefault(ordinal, []).append(log_prob)
                sum_dims.update(set(log_prob._pyro_dims))

                for frame in ordinal:
                    frame_dim = log_prob._pyro_dims[frame.dim]
                    dim_to_frame[frame_dim] = frame
                    frame_to_dim[frame] = frame_dim
                    sum_dims.remove(frame_dim)

                # Note we mark all sites with require_backward to get correct ordinals and slice non-enumerated samples
                if not node["is_observed"]:
                    queries.append(log_prob)
                    require_backward(log_prob)

        ring = SampleRing()
        contract_tensor_tree(log_probs, sum_dims, ring=ring)
        query_ordinal = {} 
        for ordinal, terms in log_probs.items():
            for term in terms:
                term._pyro_backward()
            # Note: makes collapse quadratic in number of ordinals
            for query in queries:
                if query not in query_ordinal and query._pyro_backward_result is not None:
                    query_ordinal[query] = ordinal

        collapsed_trace = OrderedDict()
        for node in enum_trace.values():
            if node["type"] == "sample" and not node["is_observed"]:
                new_node = {}
                log_prob = node["packed"]["log_prob"]
                ordinal = query_ordinal[log_prob]
                new_node["cond_indep_stack"] = tuple(
                    f for f in node["cond_indep_stack"]
                    if not f.vectorized or frame_to_dim[f] in ordinal)

                sample = log_prob._pyro_backward_result
                sample_dim = log_prob._pyro_dims[-1]
                new_node["infer"] = node["infer"].copy()
                
                new_value = node["value"]
                # TODO move this into a custom SampleRing Leaf implementation
                for index, dim in zip(sample, sample._pyro_sample_dims):
                    if dim in new_value._pyro_dims:
                        new_value = packed.gather(new_value, index, dim)
                new_node["value"] = new_value
                collapsed_trace[node["name"]] = new_node

        # TODO add observe sites induced by marginalization (one per ordinal in log_probs)
        # TODO replay model correctly against collapsed_trace (get correct cond_indep_stack)
        return model(*args, **kwargs)

    return _collapsed_model
