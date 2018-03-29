from __future__ import absolute_import, division, print_function

import math
import numbers

import torch

from pyro.distributions.util import is_identically_zero
from pyro.poutine.util import site_is_subsample


def torch_exp(x):
    """
    Like ``x.exp()`` for a :class:`~torch.Tensor`, but also accepts
    numbers.
    """
    if isinstance(x, numbers.Number):
        return math.exp(x)
    return x.exp()


def torch_data_sum(x):
    """
    Like ``x.sum().item()`` for a :class:`~torch.Tensor`, but also works
    with numbers.
    """
    if isinstance(x, numbers.Number):
        return x
    return x.sum().item()


def torch_backward(x):
    """
    Like ``x.backward()`` for a :class:`~torch.Tensor`, but also accepts
    numbers (a no-op if given a number).
    """
    if torch.is_tensor(x):
        x.backward()


def reduce_to_target(source, target):
    """
    Sums out any dimensions in source that are of size > 1 in source but of
    size 1 in target.
    """
    while source.dim() > target.dim():
        source = source.sum(0)
    for k in range(1, 1 + source.dim()):
        if source.size(-k) > target.size(-k):
            source = source.sum(-k, keepdim=True)
    return source


def reduce_to_shape(source, shape):
    """
    Sums out any dimensions in source that are of size > 1 in source but of
    size 1 in target.
    """
    while source.dim() > len(shape):
        source = source.sum(0)
    for k in range(1, 1 + source.dim()):
        if source.size(-k) > shape[-k]:
            source = source.sum(-k, keepdim=True)
    return source


def get_iarange_stacks(trace):
    """
    This builds a dict mapping site name to a set of iarange stacks.  Each
    iarange stack is a list of :class:`CondIndepStackFrame`s corresponding to
    an :class:`iarange`.  This information is used by :class:`Trace_ELBO` and
    :class:`TraceGraph_ELBO`.
    """
    return {name: [f for f in node["cond_indep_stack"] if f.vectorized]
            for name, node in trace.nodes.items()
            if node["type"] == "sample" and not site_is_subsample(node)}


class MultiFrameTensor(dict):
    """
    A container for sums of Tensors among different :class:`iarange` contexts.

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


def _dict_iadd(items, key, value):
    if key in items:
        items[key] = items[key] + value
    else:
        items[key] = value


class MultiFrameDice(object):
    """
    An implementation of the DiCE operator compatible with Pyro features.

    This implementation correctly handles:
    - scaled log-probability due to subsampling
    - independence in different contexts due to iarange
    - weights due to parallel and sequential enumeration

    This assumes restricted dependency structure on the model and guide:
    variables outside of an :class:`~pyro.iarange` can never depend on
    variables inside that :class:`~pyro.iarange`.

    Refereces:
    [1] Jakob Foerster, Greg Farquhar, Maruan Al-Shedivat, Tim Rocktaeschel,
        Eric P. Xing, Shimon Whiteson (2018)
        "DiCE: The Infinitely Differentiable Monte-Carlo Estimator"
        https://arxiv.org/abs/1802.05098
    """
    def __init__(self, guide_trace):
        log_denom = {}  # avoids double-counting when sequentially enumerating
        log_probs = {}  # accounts for upstream probabilties

        for site in guide_trace.nodes.values():
            if site["type"] != "sample":
                continue
            log_prob = site['score_parts'].score_function  # not scaled by subsampling
            if is_identically_zero(log_prob):
                continue
            context = frozenset(f for f in site["cond_indep_stack"] if f.vectorized)

            if site["infer"].get("enumerate"):
                if site["infer"]["enumerate"] == "sequential":
                    _dict_iadd(log_denom, context, math.log(site["infer"]["_enum_total"]))
            else:  # site was monte carlo sampled
                log_prob = log_prob - log_prob.detach()
            _dict_iadd(log_probs, context, log_prob)

        self.log_denom = log_denom
        self.log_probs = log_probs
        self.cache = {}

    def in_context(self, cond_indep_stack):
        """
        Returns a vectorized DiCE factor in a given :class:`~pyro.iarange` context.
        """
        target_context = frozenset(f for f in cond_indep_stack if f.vectorized)
        if target_context in self.cache:
            return self.cache[target_context]

        log_prob = 0
        for context, term in self.log_denom.items():
            if not context <= target_context:  # not downstream
                log_prob = log_prob - term  # term = log(# times this context is counted)
        for context, term in self.log_probs.items():
            if context <= target_context:  # upstream
                log_prob = log_prob + term  # term = log(dice weight of this context)
        result = 1 if is_identically_zero(log_prob) else log_prob.exp()

        self.cache[target_context] = result
        return result
