# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import numbers
from collections import Counter, defaultdict
from contextlib import contextmanager

import torch
from opt_einsum import shared_intermediates
from opt_einsum.sharing import count_cached_ops

from pyro.distributions.util import is_identically_zero
from pyro.ops import packed
from pyro.ops.einsum.adjoint import require_backward
from pyro.ops.rings import MarginalRing
from pyro.poutine.util import site_is_subsample

_VALIDATION_ENABLED = False
LAST_CACHE_SIZE = [Counter()]  # for profiling


def enable_validation(is_validate):
    global _VALIDATION_ENABLED
    _VALIDATION_ENABLED = is_validate


def is_validation_enabled():
    return _VALIDATION_ENABLED


@contextmanager
def validation_enabled(is_validate=True):
    old = is_validation_enabled()
    try:
        enable_validation(is_validate)
        yield
    finally:
        enable_validation(old)


def torch_item(x):
    """
    Like ``x.item()`` for a :class:`~torch.Tensor`, but also works with numbers.
    """
    return x if isinstance(x, numbers.Number) else x.item()


def torch_backward(x, retain_graph=None):
    """
    Like ``x.backward()`` for a :class:`~torch.Tensor`, but also accepts
    numbers and tensors without grad_fn (resulting in a no-op)
    """
    if torch.is_tensor(x) and x.grad_fn:
        x.backward(retain_graph=retain_graph)


def torch_exp(x):
    """
    Like ``x.exp()`` for a :class:`~torch.Tensor`, but also accepts
    numbers.
    """
    if torch.is_tensor(x):
        return torch.exp(x)
    else:
        return math.exp(x)


def torch_sum(tensor, dims):
    """
    Like :func:`torch.sum` but sum out dims only if they exist.
    """
    assert all(d < 0 for d in dims)
    leftmost = -tensor.dim()
    dims = [d for d in dims if leftmost <= d]
    return tensor.sum(dims) if dims else tensor


def zero_grads(tensors):
    """
    Sets gradients of list of Tensors to zero in place
    """
    for p in tensors:
        if p.grad is not None:
            p.grad = torch.zeros_like(p.grad)


def get_plate_stacks(trace):
    """
    This builds a dict mapping site name to a set of plate stacks.  Each
    plate stack is a list of :class:`CondIndepStackFrame`s corresponding to
    an :class:`plate`.  This information is used by :class:`Trace_ELBO` and
    :class:`TraceGraph_ELBO`.
    """
    return {name: [f for f in node["cond_indep_stack"] if f.vectorized]
            for name, node in trace.nodes.items()
            if node["type"] == "sample" and not site_is_subsample(node)}


def get_dependent_plate_dims(sites):
    """
    Return a list of unique dims for plates that are not common to all sites.
    """
    plate_sets = [site["cond_indep_stack"]
                  for site in sites if site["type"] == "sample"]
    all_plates = set().union(*plate_sets)
    common_plates = all_plates.intersection(*plate_sets)
    sum_plates = all_plates - common_plates
    sum_dims = sorted({f.dim for f in sum_plates if f.dim is not None})
    return sum_dims


class MultiFrameTensor(dict):
    """
    A container for sums of Tensors among different :class:`plate` contexts.

    Used in :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO` to simplify
    downstream cost computation logic.

    Example::

        downstream_cost = MultiFrameTensor()
        for site in downstream_nodes:
            downstream_cost.add((site["cond_indep_stack"], site["log_prob"]))
        downstream_cost.add(*other_costs.items())  # add in bulk
        summed = downstream_cost.sum_to(target_site["cond_indep_stack"])
    """
    def __init__(self, *items):
        super().__init__()
        self.add(*items)

    def add(self, *items):
        """
        Add a collection of (cond_indep_stack, tensor) pairs. Keys are
        ``cond_indep_stack``s, i.e. tuples of :class:`CondIndepStackFrame`s.
        Values are :class:`torch.Tensor`s.
        """
        for cond_indep_stack, value in items:
            frames = frozenset(f for f in cond_indep_stack if f.vectorized)
            assert all(f.dim < 0 and -value.dim() <= f.dim for f in frames)
            if frames in self:
                self[frames] = self[frames] + value
            else:
                self[frames] = value

    def sum_to(self, target_frames):
        total = None
        for frames, value in self.items():
            for f in frames:
                if f not in target_frames and value.shape[f.dim] != 1:
                    value = value.sum(f.dim, True)
            while value.shape and value.shape[0] == 1:
                value = value.squeeze(0)
            total = value if total is None else total + value
        return 0. if total is None else total

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ",\n\t".join([
            '({}, ...)'.format(frames) for frames in self]))


def compute_site_dice_factor(site):
    log_denom = 0
    log_prob = site["packed"]["score_parts"].score_function  # not scaled by subsampling
    dims = getattr(log_prob, "_pyro_dims", "")
    if site["infer"].get("enumerate"):
        num_samples = site["infer"].get("num_samples")
        if num_samples is not None:  # site was multiply sampled
            if not is_identically_zero(log_prob):
                log_prob = log_prob - log_prob.detach()
            log_prob = log_prob - math.log(num_samples)
            if not isinstance(log_prob, torch.Tensor):
                log_prob = torch.tensor(float(log_prob), device=site["value"].device)
            log_prob._pyro_dims = dims
            # I don't know why the following broadcast is needed, but it makes tests pass:
            log_prob, _ = packed.broadcast_all(log_prob, site["packed"]["log_prob"])
        elif site["infer"]["enumerate"] == "sequential":
            log_denom = math.log(site["infer"].get("_enum_total", num_samples))
    else:  # site was monte carlo sampled
        if not is_identically_zero(log_prob):
            log_prob = log_prob - log_prob.detach()
            log_prob._pyro_dims = dims

    return log_prob, log_denom


class Dice:
    """
    An implementation of the DiCE operator compatible with Pyro features.

    This implementation correctly handles:
    - scaled log-probability due to subsampling
    - independence in different ordinals due to plate
    - weights due to parallel and sequential enumeration
    - weights due to local multiple sampling

    This assumes restricted dependency structure on the model and guide:
    variables outside of an :class:`~pyro.plate` can never depend on
    variables inside that :class:`~pyro.plate`.

    References:
    [1] Jakob Foerster, Greg Farquhar, Maruan Al-Shedivat, Tim Rocktaeschel,
        Eric P. Xing, Shimon Whiteson (2018)
        "DiCE: The Infinitely Differentiable Monte-Carlo Estimator"
        https://arxiv.org/abs/1802.05098
    [2] Laurence Aitchison (2018)
        "Tensor Monte Carlo: particle methods for the GPU era"
        https://arxiv.org/abs/1806.08593

    :param pyro.poutine.trace.Trace guide_trace: A guide trace.
    :param ordering: A dictionary mapping model site names to ordinal values.
        Ordinal values may be any type that is (1) ``<=`` comparable and (2)
        hashable; the canonical ordinal is a ``frozenset`` of site names.
    """
    def __init__(self, guide_trace, ordering):
        log_denoms = defaultdict(float)  # avoids double-counting when sequentially enumerating
        log_probs = defaultdict(list)  # accounts for upstream probabilties

        for name, site in guide_trace.nodes.items():
            if site["type"] != "sample":
                continue

            ordinal = ordering[name]
            log_prob, log_denom = compute_site_dice_factor(site)
            if not is_identically_zero(log_prob):
                log_probs[ordinal].append(log_prob)
            if not is_identically_zero(log_denom):
                log_denoms[ordinal] += log_denom

        self.log_denom = log_denoms
        self.log_probs = log_probs

    def _get_log_factors(self, target_ordinal):
        """
        Returns a list of DiCE factors at a given ordinal.
        """
        log_denom = 0
        for ordinal, term in self.log_denom.items():
            if not ordinal <= target_ordinal:  # not downstream
                log_denom += term  # term = log(# times this ordinal is counted)

        log_factors = [] if is_identically_zero(log_denom) else [-log_denom]
        for ordinal, terms in self.log_probs.items():
            if ordinal <= target_ordinal:  # upstream
                log_factors.extend(terms)  # terms = [log(dice weight of this ordinal)]

        return log_factors

    def compute_expectation(self, costs):
        """
        Returns a differentiable expected cost, summing over costs at given ordinals.

        :param dict costs: A dict mapping ordinals to lists of cost tensors
        :returns: a scalar expected cost
        :rtype: torch.Tensor or float
        """
        # Share computation across all cost terms.
        with shared_intermediates() as cache:
            ring = MarginalRing(cache=cache)
            expected_cost = 0.
            for ordinal, cost_terms in costs.items():
                log_factors = self._get_log_factors(ordinal)
                scale = math.exp(sum(x for x in log_factors if not isinstance(x, torch.Tensor)))
                log_factors = [x for x in log_factors if isinstance(x, torch.Tensor)]

                # Collect log_prob terms to query for marginal probability.
                queries = {frozenset(cost._pyro_dims): None for cost in cost_terms}
                for log_factor in log_factors:
                    key = frozenset(log_factor._pyro_dims)
                    if queries.get(key, False) is None:
                        queries[key] = log_factor
                # Ensure a query exists for each cost term.
                for cost in cost_terms:
                    key = frozenset(cost._pyro_dims)
                    if queries[key] is None:
                        query = torch.zeros_like(cost)
                        query._pyro_dims = cost._pyro_dims
                        log_factors.append(query)
                        queries[key] = query

                # Perform sum-product contraction. Note that plates never need to be
                # product-contracted due to our plate-based dependency ordering.
                sum_dims = set().union(*(x._pyro_dims for x in log_factors)) - ordinal
                for query in queries.values():
                    require_backward(query)
                root = ring.sumproduct(log_factors, sum_dims)
                root._pyro_backward()
                probs = {key: query._pyro_backward_result.exp() for key, query in queries.items()}

                # Aggregate prob * cost terms.
                for cost in cost_terms:
                    key = frozenset(cost._pyro_dims)
                    prob = probs[key]
                    prob._pyro_dims = queries[key]._pyro_dims
                    mask = prob > 0
                    if torch._C._get_tracing_state() or not mask.all():
                        mask._pyro_dims = prob._pyro_dims
                        cost, prob, mask = packed.broadcast_all(cost, prob, mask)
                        prob = prob.masked_select(mask)
                        cost = cost.masked_select(mask)
                    else:
                        cost, prob = packed.broadcast_all(cost, prob)
                    expected_cost = expected_cost + scale * torch.tensordot(prob, cost, prob.dim())

        LAST_CACHE_SIZE[0] = count_cached_ops(cache)
        return expected_cost


def check_fully_reparametrized(guide_site):
    log_prob, score_function_term, entropy_term = guide_site["score_parts"]
    fully_rep = (guide_site["fn"].has_rsample and not is_identically_zero(entropy_term) and
                 is_identically_zero(score_function_term))
    if not fully_rep:
        raise NotImplementedError("All distributions in the guide must be fully reparameterized.")
