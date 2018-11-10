from __future__ import absolute_import, division, print_function

import math
import numbers
from collections import Counter, defaultdict

import torch
from opt_einsum import shared_intermediates
from opt_einsum.sharing import count_cached_ops

from pyro.distributions.util import is_identically_zero
from pyro.ops import packed
from pyro.poutine.util import site_is_subsample

_VALIDATION_ENABLED = False
LAST_CACHE_SIZE = [Counter()]  # for profiling


def enable_validation(is_validate):
    global _VALIDATION_ENABLED
    _VALIDATION_ENABLED = is_validate


def is_validation_enabled():
    return _VALIDATION_ENABLED


def torch_item(x):
    """
    Like ``x.item()`` for a :class:`~torch.Tensor`, but also works with numbers.
    """
    return x if isinstance(x, numbers.Number) else x.item()


def torch_backward(x, retain_graph=None):
    """
    Like ``x.backward()`` for a :class:`~torch.Tensor`, but also accepts
    numbers (a no-op if given a number).
    """
    if torch.is_tensor(x):
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


def detach_iterable(iterable):
    if torch.is_tensor(iterable):
        return iterable.detach()
    else:
        return [var.detach() for var in iterable]


def zero_grads(tensors):
    """
    Sets gradients of list of Tensors to zero in place
    """
    for p in tensors:
        if p.grad is not None:
            p.grad = p.grad.new_zeros(p.shape)


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
        super(MultiFrameTensor, self).__init__()
        self.add(*items)

    def add(self, *items):
        """
        Add a collection of (cond_indep_stack, tensor) pairs. Keys are
        ``cond_indep_stack``s, i.e. tuples of :class:`CondIndepStackFrame`s.
        Values are :class:`torch.Tensor`s.
        """
        for cond_indep_stack, value in items:
            frames = frozenset(f for f in cond_indep_stack if f.vectorized)
            assert all(f.dim < 0 and -len(value.shape) <= f.dim for f in frames)
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
                value.squeeze_(0)
            total = value if total is None else total + value
        return total

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ",\n\t".join([
            '({}, ...)'.format(frames) for frames in self]))


class Dice(object):
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
        log_denom = defaultdict(float)  # avoids double-counting when sequentially enumerating
        log_probs = defaultdict(list)  # accounts for upstream probabilties

        for name, site in guide_trace.nodes.items():
            if site["type"] != "sample":
                continue

            log_prob = site["packed"]["score_parts"].score_function  # not scaled by subsampling
            dims = getattr(log_prob, "_pyro_dims", "")
            ordinal = ordering[name]
            if site["infer"].get("enumerate"):
                num_samples = site["infer"].get("num_samples")
                if num_samples is not None:  # site was multiply sampled
                    if not is_identically_zero(log_prob):
                        log_prob = log_prob - log_prob.detach()
                    log_prob = log_prob - math.log(num_samples)
                    if not isinstance(log_prob, torch.Tensor):
                        log_prob = site["value"].new_tensor(log_prob)
                    log_prob._pyro_dims = dims
                elif site["infer"]["enumerate"] == "sequential":
                    log_denom[ordinal] += math.log(site["infer"]["_enum_total"])
            else:  # site was monte carlo sampled
                if is_identically_zero(log_prob):
                    continue
                log_prob = log_prob - log_prob.detach()
                log_prob._pyro_dims = dims
            log_probs[ordinal].append(log_prob)

        self.log_denom = log_denom
        self.log_probs = log_probs
        self._log_factors_cache = {}
        self._prob_cache = {}

    def _get_log_factors(self, target_ordinal):
        """
        Returns a list of DiCE factors at a given ordinal.
        """
        # memoize
        try:
            return self._log_factors_cache[target_ordinal]
        except KeyError:
            pass

        log_denom = 0
        for ordinal, term in self.log_denom.items():
            if not ordinal <= target_ordinal:  # not downstream
                log_denom += term  # term = log(# times this ordinal is counted)

        log_factors = [] if is_identically_zero(log_denom) else [-log_denom]
        for ordinal, terms in self.log_probs.items():
            if ordinal <= target_ordinal:  # upstream
                log_factors.extend(terms)  # terms = [log(dice weight of this ordinal)]

        self._log_factors_cache[target_ordinal] = log_factors
        return log_factors

    def compute_expectation(self, costs):
        """
        Returns a differentiable expected cost, summing over costs at given ordinals.

        :param dict costs: A dict mapping ordinals to lists of cost tensors
        :returns: a scalar expected cost
        :rtype: torch.Tensor or float
        """
        # precompute exponentials to be shared across calls to sumproduct
        exp_table = {}
        factors_table = defaultdict(list)
        for ordinal in costs:
            for log_factor in self._get_log_factors(ordinal):
                key = id(log_factor)
                if key in exp_table:
                    factor = exp_table[key]
                else:
                    factor = packed.exp(log_factor)
                    exp_table[key] = factor
                factors_table[ordinal].append(factor)

        # share computation across all cost terms
        with shared_intermediates() as cache:
            expected_cost = 0.
            for ordinal, cost_terms in costs.items():
                factors = factors_table.get(ordinal, [])
                for cost in cost_terms:
                    dims = ''.join(dim for dim in cost._pyro_dims
                                   if any(dim in getattr(f, '_pyro_dims', '') for f in factors))
                    prob = packed.sumproduct(factors, dims, device=cost.device)
                    mask = prob > 0
                    mask._pyro_dims = prob._pyro_dims
                    if torch.is_tensor(mask) and not mask.all():
                        cost, prob, mask = packed.broadcast_all(cost, prob, mask)
                        prob = prob[mask]
                        cost = cost[mask]
                        expected_cost = expected_cost + (prob * cost).sum()
                    else:
                        expected_cost = expected_cost + packed.sumproduct([prob, cost])
        LAST_CACHE_SIZE[0] = count_cached_ops(cache)
        return expected_cost
