from __future__ import absolute_import, division, print_function

import numbers

import torch
from torch.autograd import Variable

from pyro.distributions.util import sum_leftmost


def torch_data_sum(x):
    """
    Like ``x.data.sum()`` for a ``torch.autograd.Variable``, but also works
    with numbers.
    """
    if isinstance(x, numbers.Number):
        return x
    return x.data.sum()


def torch_sum(x):
    """
    Like ``x.sum()`` for a ``torch.autograd.Variable``, but also works with
    numbers.
    """
    if isinstance(x, numbers.Number):
        return x
    return x.sum()


def torch_backward(x):
    """
    Like ``x.backward()`` for a ``torch.autograd.Variable``, but also accepts
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


class MultiViewTensor(dict):
    """
    A container for Variables with different shapes.

    Used in :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO` to simplify
    downstream cost computation logic.

    Example::

        downstream_cost = MultiViewTensor()
        downstream_cost.add(self.cost)
        for node in downstream_nodes:
            summed = node.downstream_cost.sum_leftmost(dims)
            downstream_cost.add(summed)
    """
    def __init__(self, value=None):
        if value is not None:
            if isinstance(value, Variable):
                self[value.shape] = value

    def add(self, term):
        """
        Add tensor to collection of tensors stored in MultiViewTensor.
        key by shape.
        """
        if isinstance(term, Variable):
            if term.shape in self:
                self[term.shape] = self[term.shape] + term
            else:
                self[term.shape] = term
        else:
            for shape, value in term.items():
                if shape in self:
                    self[shape] = self[shape] + value
                else:
                    self[shape] = value

    def sum_leftmost_all_but(self, dim):
        """
        This behaves like ``sum_leftmost(term, -dim)`` except for dim=0 where
        everything is summed out.
        """
        assert dim >= 0
        result = MultiViewTensor()
        for shape, term in self.items():
            if dim == 0:
                result.add(term.sum())
            elif dim > term.dim():
                result.add(term)
            else:
                result.add(sum_leftmost(term, -dim))
        return result

    def contract_as(self, target):
        """Opposite of :meth:`torch.Tensor.expand_as`."""
        if not self:
            return 0
        return sum(reduce_to_target(x, target) for x in self.values())

    def contract(self, shape):
        """Opposite of  :meth:`torch.Tensor.expand`."""
        if not self:
            return 0
        return sum(reduce_to_shape(x, shape) for x in self.values())

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ", ".join([str(k) for k in self.keys()]))


class TensorTree(object):
    def __init__(self):
        self._terms = {}
        self._upstream = {}
        self._frozen = False

    def add(self, key, value):
        """
        Adds a term at one node.
        """
        assert not self._frozen, 'Cannot call TensorTree.add() after .get_upstream()'
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
            self._frozen = True
            result = self._terms.get(key)
            if key:
                upstream = self.get_upstream(key[:-1])
                if upstream is not None:
                    result = upstream if result is None else result + upstream
            self._upstream[key] = result
            return result

    def prune_upstream(self, key):
        """
        Prunes a key and all upstream keys.
        """
        # First save upstream costs
        for key2 in self._terms:
            self.get_upstream(key2)

        # Then delete this and all upstream keys.
        del self._terms[key]
        self._upstream[key] = None
        while key:
            key = key[:-1]
            del self._terms[key]
            self._upstream[key] = None

    def exp(self):
        # First save upstream costs
        for key2 in self._terms:
            self.get_upstream(key2)

        # Then create an exponentiated copy of _upstream.
        result = TensorTree()
        if self._upstream:
            for key, term in self._upstream.items():
                result._upstream[key] = None if term is None else term.exp()
        else:
            result._upstream[()] = 1
        result._frozen = True

        return result
