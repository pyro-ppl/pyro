# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
import warnings
from collections import OrderedDict

from opt_einsum import shared_intermediates

import pyro.ops.packed as packed
from pyro import poutine
from pyro.infer.traceenum_elbo import TraceEnum_ELBO
from pyro.ops.contract import contract_tensor_tree
from pyro.ops.einsum.adjoint import require_backward
from pyro.ops.rings import MapRing, SampleRing
from pyro.poutine.enum_messenger import EnumMessenger
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.util import prune_subsample_sites
from pyro.util import jit_iter

_RINGS = {0: MapRing, 1: SampleRing}


def _make_ring(temperature, cache, dim_to_size):
    try:
        return _RINGS[temperature](cache=cache, dim_to_size=dim_to_size)
    except KeyError as e:
        raise ValueError("temperature must be 0 (map) or 1 (sample) for now") from e


class SamplePosteriorMessenger(ReplayMessenger):
    # This acts like ReplayMessenger but additionally replays cond_indep_stack.

    def _pyro_sample(self, msg):
        if msg["infer"].get("enumerate") == "parallel":
            super()._pyro_sample(msg)
        if msg["name"] in self.trace:
            msg["cond_indep_stack"] = self.trace.nodes[msg["name"]]["cond_indep_stack"]


def _sample_posterior(model, first_available_dim, temperature, strict_enumeration_warning,
                      *args, **kwargs):
    # For internal use by infer_discrete.

    # Create an enumerated trace.
    with poutine.block(), EnumMessenger(first_available_dim):
        enum_trace = poutine.trace(model).get_trace(*args, **kwargs)
    enum_trace = prune_subsample_sites(enum_trace)
    enum_trace.compute_log_prob()
    enum_trace.pack_tensors()

    return _sample_posterior_from_trace(model, enum_trace, temperature,
                                        strict_enumeration_warning, *args, **kwargs)


def _sample_posterior_from_trace(model, enum_trace, temperature, strict_enumeration_warning,
                                 *args, **kwargs):
    plate_to_symbol = enum_trace.plate_to_symbol

    # Collect a set of query sample sites to which the backward algorithm will propagate.
    sum_dims = set()
    queries = []
    dim_to_size = {}
    cost_terms = OrderedDict()
    enum_terms = OrderedDict()
    for node in enum_trace.nodes.values():
        if node["type"] == "sample":
            ordinal = frozenset(plate_to_symbol[f.name]
                                for f in node["cond_indep_stack"]
                                if f.vectorized and f.size > 1)
            # For sites that depend on an enumerated variable, we need to apply
            # the mask but not the scale when sampling.
            if "masked_log_prob" not in node["packed"]:
                node["packed"]["masked_log_prob"] = packed.scale_and_mask(
                    node["packed"]["unscaled_log_prob"], mask=node["packed"]["mask"])
            log_prob = node["packed"]["masked_log_prob"]
            sum_dims.update(frozenset(log_prob._pyro_dims) - ordinal)
            if sum_dims.isdisjoint(log_prob._pyro_dims):
                continue
            dim_to_size.update(zip(log_prob._pyro_dims, log_prob.shape))
            if node["infer"].get("_enumerate_dim") is None:
                cost_terms.setdefault(ordinal, []).append(log_prob)
            else:
                enum_terms.setdefault(ordinal, []).append(log_prob)
            # Note we mark all sample sites with require_backward to gather
            # enumerated sites and adjust cond_indep_stack of all sample sites.
            if not node["is_observed"]:
                queries.append(log_prob)
                require_backward(log_prob)

    if strict_enumeration_warning and not enum_terms:
        warnings.warn('infer_discrete found no sample sites configured for enumeration. '
                      'If you want to enumerate sites, you need to @config_enumerate or set '
                      'infer={"enumerate": "sequential"} or infer={"enumerate": "parallel"}.')

    # We take special care to match the term ordering in
    # pyro.infer.traceenum_elbo._compute_model_factors() to allow
    # contract_tensor_tree() to use shared_intermediates() inside
    # TraceEnumSample_ELBO. The special ordering is: first all cost terms in
    # order of model_trace, then all enum_terms in order of model trace.
    log_probs = cost_terms
    for ordinal, terms in enum_terms.items():
        log_probs.setdefault(ordinal, []).extend(terms)

    # Run forward-backward algorithm, collecting the ordinal of each connected component.
    cache = getattr(enum_trace, "_sharing_cache", {})
    ring = _make_ring(temperature, cache, dim_to_size)
    with shared_intermediates(cache):
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
            log_prob = node["packed"]["masked_log_prob"]
            if hasattr(log_prob, "_pyro_backward_result"):
                # Adjust the cond_indep_stack.
                ordinal = query_to_ordinal[log_prob]
                new_node["cond_indep_stack"] = tuple(
                    f for f in node["cond_indep_stack"]
                    if not (f.vectorized and f.size > 1) or plate_to_symbol[f.name] in ordinal)

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


def infer_discrete(fn=None, first_available_dim=None, temperature=1, *,
                   strict_enumeration_warning=True):
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
    :param bool strict_enumeration_warning: Whether to warn in case no
        enumerated sample sites are found. Defalts to True.
    """
    assert first_available_dim < 0, first_available_dim
    if fn is None:  # support use as a decorator
        return functools.partial(infer_discrete,
                                 first_available_dim=first_available_dim,
                                 temperature=temperature)
    return functools.partial(_sample_posterior, fn, first_available_dim, temperature,
                             strict_enumeration_warning)


class TraceEnumSample_ELBO(TraceEnum_ELBO):
    """
    This extends :class:`TraceEnum_ELBO` to make it cheaper to sample from
    discrete latent states during SVI.

    The following are equivalent but the first is cheaper, sharing work
    between the computations of ``loss`` and ``z``::

        # Version 1.
        elbo = TraceEnumSample_ELBO(max_plate_nesting=1)
        loss = elbo.loss(*args, **kwargs)
        z = elbo.sample_saved()

        # Version 2.
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        loss = elbo.loss(*args, **kwargs)
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        z = infer_discrete(poutine.replay(model, guide_trace),
                           first_available_dim=-2)(*args, **kwargs)

    """
    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(
            model, guide, args, kwargs)

        # Mark all sample sites with require_backward to gather enumerated
        # sites and adjust cond_indep_stack of all sample sites.
        for node in model_trace.nodes.values():
            if node["type"] == "sample" and not node["is_observed"]:
                log_prob = node["packed"]["unscaled_log_prob"]
                require_backward(log_prob)

        self._saved_state = model, model_trace, guide_trace, args, kwargs
        return model_trace, guide_trace

    def sample_saved(self):
        """
        Generate latent samples while reusing work from SVI.step().
        """
        model, model_trace, guide_trace, args, kwargs = self._saved_state
        model = poutine.replay(model, guide_trace)
        temperature = 1
        return _sample_posterior_from_trace(model, model_trace, temperature,
                                            self.strict_enumeration_warning,
                                            *args, **kwargs)
