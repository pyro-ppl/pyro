from __future__ import absolute_import, division, print_function

import math
import numbers

import torch

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
    if isinstance(x, torch.autograd.Variable):
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
            downstream_cost.add((site["cond_indep_stack"], site["batch_log_pdf"]))
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


class TreeSum(object):
    """
    Data structure to compute cumulative costs along paths in a tree.
    Typically keys are ``cond_indep_stack``s.
    """
    def __init__(self):
        self._terms = {}
        self._upstream = {}
        self._frozen = False

    def copy(self):
        result = TreeSum()
        result._terms = self._terms.copy()
        result._upstream = self._upstream.copy()
        result._frozen = self._frozen
        return result

    def add(self, key, value):
        """
        Adds a term at one node.
        """
        assert not self._frozen, 'Cannot call TreeSum.add() after .get_upstream()'
        if key in self._terms:
            self._terms[key] = self._terms[key] + value
        else:
            self._terms[key] = value

    def get_upstream(self, key):
        """
        Returns upstream sum or None. None denotes zero.
        """
        try:
            return self._upstream[key]
        except KeyError:
            result = self._terms.get(key)
            if key:
                upstream = self.get_upstream(key[:-1])
                if upstream is not None:
                    result = upstream if result is None else upstream + result
            self._upstream[key] = result
            self._frozen = True
            return result

    def _freeze(self):
        for key in self._terms:
            self.get_upstream(key)
        self._frozen = True

    def items(self):
        self._freeze()
        return self._upstream.items()

    def exp(self):
        self._freeze()
        # Exponentiate _only_ the supporting terms of self._upstream.
        # This restriction is required for .prune() to work correctly.
        result = TreeSum()
        result._upstream = {key: torch_exp(self._upstream[key]) for key in self._terms}
        result._frozen = True
        return result

    def prune(self, key):
        assert self._frozen, 'Cannot call TreeSum.prune() before freezing'
        self._upstream.pop(key, None)
        self._terms.pop(key, None)
