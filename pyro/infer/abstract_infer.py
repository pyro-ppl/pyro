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

        d = dist.Categorical(logits=logits, one_hot=False)
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


class Marginal(Histogram):
    """
    :param trace_dist: a TracePosterior instance representing a Monte Carlo posterior

    Marginal histogram distribution.
    Turns a TracePosterior object into a Distribution
    over the return values of the TracePosterior's model.
    """
    def __init__(self, trace_dist, sites=None):
        assert isinstance(trace_dist, TracePosterior), \
            "trace_dist must be trace posterior distribution object"

        if sites is None:
            sites = "_RETURN"

        assert isinstance(sites, (str, list)), \
            "sites must be either '_RETURN' or list"

        if isinstance(sites, str):
            assert sites in ("_RETURN",), \
                "sites string must be '_RETURN'"

        self.sites = sites
        super(Marginal, self).__init__()
        self.trace_dist = trace_dist

    def _gen_weighted_samples(self, *args, **kwargs):
        for tr, log_w in poutine.block(self.trace_dist._traces)(*args, **kwargs):
            if self.sites == "_RETURN":
                val = tr.nodes["_RETURN"]["value"]
            else:
                val = {name: tr.nodes[name]["value"]
                       for name in self.sites}
            yield (val, log_w)

    def batch_log_pdf(self, val, *args, **kwargs):
        raise NotImplementedError("batch_log_pdf not well defined for Marginal")


class TracePosterior(object):
    """
    Abstract TracePosterior object from which posterior inference algorithms inherit.
    Holds a generator over Traces sampled from the approximate posterior.
    Not actually a distribution object - no sample or score methods.
    """
    def __init__(self):
        pass

    def _traces(self, *args, **kwargs):
        """
        Abstract method.
        Get unnormalized weighted list of posterior traces
        """
        raise NotImplementedError("inference algorithm must implement _traces")

    def __call__(self, *args, **kwargs):
        traces, logits = [], []
        for tr, logit in poutine.block(self._traces)(*args, **kwargs):
            traces.append(tr)
            logits.append(logit)
        logits = torch.stack(logits).squeeze()
        logits -= util.log_sum_exp(logits)
        if not isinstance(logits, torch.autograd.Variable):
            logits = Variable(logits)
        ix = dist.categorical(logits=logits, one_hot=False)
        return traces[ix.data[0]]
