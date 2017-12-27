from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.util as util


def _eq(x, y):
    """
    Equality comparison for nested data structures with tensors.
    """
    if type(x) is not type(y):
        return False
    elif isinstance(x, dict):
        if set(x.keys()) != set(y.keys()):
            return False
        return all(_eq(x_val, y[key]) for key, x_val in x.items())
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        return (x == y).all()
    elif isinstance(x, torch.autograd.Variable):
        return (x.data == y.data).all()
    else:
        return x == y


def _index(seq, value):
    """
    Find position of ``value`` in ``seq`` using ``_eq`` to test equality.
    Returns ``-1`` if ``value`` is not in ``seq``.
    """
    for i, x in enumerate(seq):
        if _eq(x, value):
            return i
    return -1


class Histogram(dist.Distribution):
    """
    Abstract Histogram distribution of equality-comparable values.
    Should only be used inside Marginal.
    """
    enumerable = True

    @util.memoize
    def _dist_and_values(self, *args, **kwargs):
        # XXX currently this whole object is very inefficient
        values, logits = [], []
        for value, logit in self._gen_weighted_samples(*args, **kwargs):
            ix = _index(values, value)
            if ix == -1:
                # Value is new.
                values.append(value)
                logits.append(logit)
            else:
                # Value has already been seen.
                logits[ix] = util.log_sum_exp(torch.stack([logits[ix], logit]).squeeze())

        logits = torch.stack(logits).squeeze()
        logits -= util.log_sum_exp(logits)
        if not isinstance(logits, torch.autograd.Variable):
            logits = Variable(logits)
        logits = logits - util.log_sum_exp(logits)

        d = dist.Categorical(logits=logits)
        return d, values

    def _gen_weighted_samples(self, *args, **kwargs):
        raise NotImplementedError("_gen_weighted_samples is abstract method")

    def sample(self, *args, **kwargs):
        d, values = self._dist_and_values(*args, **kwargs)
        ix = d.sample().data[0]
        return values[ix]

    def log_pdf(self, val, *args, **kwargs):
        d, values = self._dist_and_values(*args, **kwargs)
        ix = _index(values, val)
        return d.log_pdf(Variable(torch.Tensor([ix])))

    def batch_log_pdf(self, val, *args, **kwargs):
        d, values = self._dist_and_values(*args, **kwargs)
        ix = _index(values, val)
        return d.batch_log_pdf(Variable(torch.Tensor([ix])))

    def enumerate_support(self, *args, **kwargs):
        d, values = self._dist_and_values(*args, **kwargs)
        return values[:]
