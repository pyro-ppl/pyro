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
    Sums out any dimensions in source that are of size > 1 in source but of size 1 in target.
    This preserves source.dim().
    """
    for k in range(1, 1 + source.dim()):
        if k > target.dim() or source.size(-k) > target.size(-k):
            source = source.sum(-k, keepdim=True)
    return source


class MultiViewTensor(dict):
    """
    A container for Variables with different shapes. Used in TraceGraph_ELBO
    to simplify downstream cost computation logic.

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
        Add tensor to collection of tensors stored in MultiViewTensor; key by shape
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
        This behaves like sum_leftmost(term, -dim) except for dim=0 where everything is summed out
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

    def contract_to(self, target):
        """Opposite of broadcast."""
        result = 0
        for tensor in self.values():
            result = result + reduce_to_target(tensor, target)
        return result

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ", ".join([str(k) for k in self.keys()]))
