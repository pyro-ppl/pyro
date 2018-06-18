from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.abstract_infer import TracePosterior


def _dict_to_tuple(d):
    """
    Recursively converts a dictionary to a list of key-value tuples
    Only intended for use as a helper function inside memoize!!
    May break when keys cant be sorted, but that is not an expected use-case
    """
    if isinstance(d, dict):
        return tuple([(k, _dict_to_tuple(d[k])) for k in sorted(d.keys())])
    else:
        return d


def memoize(fn):
    """
    https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python
    unbounded memoize
    alternate in py3: https://docs.python.org/3/library/functools.html
    lru_cache
    """
    mem = {}

    def _fn(*args, **kwargs):
        kwargs_tuple = _dict_to_tuple(kwargs)
        if (args, kwargs_tuple) not in mem:
            mem[(args, kwargs_tuple)] = fn(*args, **kwargs)
        return mem[(args, kwargs_tuple)]
    return _fn


class HashingMarginal(dist.Distribution):
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
        super(HashingMarginal, self).__init__()
        self.trace_dist = trace_dist

    has_enumerate_support = True

    # @memoize
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
                logits[value_hash] = dist.util.log_sum_exp(torch.stack([logits[value_hash], logit]))
            else:
                logits[value_hash] = logit
                values_map[value_hash] = value

        logits = torch.stack(list(logits.values())).contiguous().view(-1)
        logits -= dist.util.log_sum_exp(logits)
        logits = logits - dist.util.log_sum_exp(logits)
        d = dist.Categorical(logits=logits)
        return d, values_map

    def _gen_weighted_samples(self, *args, **kwargs):
        for tr, log_w in poutine.block(self.trace_dist._traces)(*args, **kwargs):
            if self.sites == "_RETURN":
                val = tr.nodes["_RETURN"]["value"]
            else:
                val = {name: tr.nodes[name]["value"]
                       for name in self.sites}
            yield (val, log_w)

    def sample(self, *args, **kwargs):
        sample_shape = kwargs.pop("sample_shape", None)
        if sample_shape:
            raise ValueError("Arbitrary `sample_shape` not supported by Histogram class.")
        d, values_map = self._dist_and_values(*args, **kwargs)
        ix = d.sample()
        return list(values_map.values())[ix]

    def log_prob(self, val, *args, **kwargs):
        d, values_map = self._dist_and_values(*args, **kwargs)
        if torch.is_tensor(val):
            value_hash = hash(val.cpu().contiguous().numpy().tobytes())
        else:
            value_hash = hash(val)
        return d.log_prob(torch.tensor([values_map.keys().index(value_hash)]))

    def enumerate_support(self, *args, **kwargs):
        d, values_map = self._dist_and_values(*args, **kwargs)
        return list(values_map.values())[:]
