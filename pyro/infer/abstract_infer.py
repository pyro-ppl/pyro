from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
from torch.autograd import Variable

import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.util as util


class Histogram(dist.Distribution):
    """
    Abstract Histogram distribution of equality-comparable values.
    Should only be used inside Marginal.
    """
    enumerable = True

    @util.memoize
    def _dist_and_values(self, *args, **kwargs):
        # XXX currently this whole object is very inefficient
        values_map, logits = OrderedDict(), OrderedDict()
        for value, logit in self._gen_weighted_samples(*args, **kwargs):
            if torch.is_tensor(value):
                value_hash = hash(value.cpu().contiguous().numpy().tobytes())
            else:
                value_hash = hash(value)
            if value_hash in logits:
                # Value has already been seen.
                logits[value_hash] = util.log_sum_exp(torch.stack([logits[value_hash], logit]))
            else:
                logits[value_hash] = logit
                values_map[value_hash] = value

        logits = torch.stack(list(logits.values())).contiguous().view(-1)
        logits -= util.log_sum_exp(logits)
        if not isinstance(logits, torch.autograd.Variable):
            logits = Variable(logits)
        logits = logits - util.log_sum_exp(logits)
        d = dist.Categorical(logits=logits)
        return d, values_map

    def _gen_weighted_samples(self, *args, **kwargs):
        raise NotImplementedError("_gen_weighted_samples is abstract method")

    def sample(self, *args, **kwargs):
        sample_shape = kwargs.pop("sample_shape", None)
        if sample_shape:
            raise ValueError("Arbitrary `sample_shape` not supported by Histogram class.")
        d, values_map = self._dist_and_values(*args, **kwargs)
        ix = d.sample()
        return list(values_map.values())[ix]

    __call__ = sample

    def log_prob(self, val, *args, **kwargs):
        d, values_map = self._dist_and_values(*args, **kwargs)
        if torch.is_tensor(val):
            value_hash = hash(val.cpu().contiguous().numpy().tobytes())
        else:
            value_hash = hash(val)
        return d.log_prob(Variable(torch.Tensor([values_map.keys().index(value_hash)])))

    def enumerate_support(self, *args, **kwargs):
        d, values_map = self._dist_and_values(*args, **kwargs)
        return list(values_map.values())[:]


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
        ix = dist.Categorical(logits=logits).sample()
        return traces[ix]
