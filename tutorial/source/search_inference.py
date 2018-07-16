"""
Inference algorithms and utilities used in the RSA example models.

Adapted from: http://dippl.org/chapters/03-enumeration.html
"""

from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.infer.abstract_infer import TracePosterior
from pyro.poutine.runtime import NonlocalExit

import six
from six.moves import queue
import collections
if six.PY3:
    import functools
else:
    import functools32 as functools


def memoize(fn=None, **kwargs):
    if fn is None:
        return lambda _fn: memoize(_fn, **kwargs)
    return functools.lru_cache(**kwargs)(fn)


def factor(name, value):
    """
    Like factor in webPPL, adds a scalar weight to the log-probability of the trace
    """
    value = value if torch.is_tensor(value) else torch.tensor(value)
    d = dist.Bernoulli(logits=value)
    pyro.sample(name, d, obs=torch.ones(value.size()))


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

        self.sites = sites
        super(HashingMarginal, self).__init__()
        self.trace_dist = trace_dist

    has_enumerate_support = True

    @memoize(maxsize=10)
    def _dist_and_values(self):
        # XXX currently this whole object is very inefficient
        values_map, logits = collections.OrderedDict(), collections.OrderedDict()
        for tr, logit in zip(self.trace_dist.exec_traces,
                             self.trace_dist.log_weights):
            if isinstance(self.sites, str):
                value = tr.nodes[self.sites]["value"]
            else:
                value = {site: tr.nodes[site]["value"] for site in self.sites}
            if not torch.is_tensor(logit):
                logit = torch.tensor(logit)

            if torch.is_tensor(value):
                value_hash = hash(value.cpu().contiguous().numpy().tobytes())
            elif isinstance(value, dict):
                value_hash = hash(self._dict_to_tuple(value))
            else:
                value_hash = hash(value)
            if value_hash in logits:
                # Value has already been seen.
                logits[value_hash] = dist.util.log_sum_exp(torch.stack([logits[value_hash], logit]))
            else:
                logits[value_hash] = logit
                values_map[value_hash] = value

        logits = torch.stack(list(logits.values())).contiguous().view(-1)
        logits = logits - dist.util.log_sum_exp(logits)
        d = dist.Categorical(logits=logits)
        return d, values_map

    def sample(self):
        d, values_map = self._dist_and_values()
        ix = d.sample()
        return list(values_map.values())[ix]

    def log_prob(self, val):
        d, values_map = self._dist_and_values()
        if torch.is_tensor(val):
            value_hash = hash(val.cpu().contiguous().numpy().tobytes())
        elif isinstance(val, dict):
            value_hash = hash(self._dict_to_tuple(val))
        else:
            value_hash = hash(val)
        return d.log_prob(torch.tensor([list(values_map.keys()).index(value_hash)]))

    def enumerate_support(self):
        d, values_map = self._dist_and_values()
        return list(values_map.values())[:]

    def _dict_to_tuple(self, d):
        """
        Recursively converts a dictionary to a list of key-value tuples
        Only intended for use as a helper function inside HashingMarginal!!
        May break when keys cant be sorted, but that is not an expected use-case
        """
        if isinstance(d, dict):
            return tuple([(k, self._dict_to_tuple(d[k])) for k in sorted(d.keys())])
        else:
            return d

    def _weighted_mean(self, value, dim=0):
        weights = self._dist_and_values()[0].logits
        for _ in range(value.dim() - 1):
            weights = weights.unsqueeze(-1)
        max_val = weights.max(dim)[0]
        return max_val.exp() * (value * (weights - max_val.unsqueeze(-1)).exp()).sum(dim=dim)

    @property
    def mean(self):
        samples = torch.stack(list(self._dist_and_values()[1].values()))
        return self._weighted_mean(samples) / self._weighted_mean(samples.new_tensor([1.]))

    @property
    def variance(self):
        samples = torch.stack(list(self._dist_and_values()[1].values()))
        deviation_squared = torch.pow(samples - self.mean, 2)
        return self._weighted_mean(deviation_squared) / self._weighted_mean(samples.new_tensor([1.]))


########################
# Exact Search inference
########################

class Search(TracePosterior):
    """
    Exact inference by enumerating over all possible executions
    """
    def __init__(self, model, max_tries=int(1e6), **kwargs):
        self.model = model
        self.max_tries = max_tries
        super(Search, self).__init__(**kwargs)

    def _traces(self, *args, **kwargs):
        q = queue.Queue()
        q.put(poutine.Trace())
        p = poutine.trace(
            poutine.queue(self.model, queue=q, max_tries=self.max_tries))
        while not q.empty():
            tr = p.get_trace(*args, **kwargs)
            yield tr, tr.log_prob_sum()


###############################################
# Best-first Search Inference
###############################################


def pqueue(fn, queue):

    def sample_escape(tr, site):
        return (site["name"] not in tr) and \
            (site["type"] == "sample") and \
            (not site["is_observed"])

    def _fn(*args, **kwargs):

        for i in range(int(1e6)):
            assert not queue.empty(), \
                "trying to get() from an empty queue will deadlock"

            priority, next_trace = queue.get()
            try:
                ftr = poutine.trace(poutine.escape(poutine.replay(fn, next_trace),
                                                   functools.partial(sample_escape,
                                                                     next_trace)))
                return ftr(*args, **kwargs)
            except NonlocalExit as site_container:
                site_container.reset_stack()
                for tr in poutine.util.enum_extend(ftr.trace.copy(),
                                                   site_container.site):
                    # add a little bit of noise to the priority to break ties...
                    queue.put((tr.log_prob_sum().item() - torch.rand(1).item() * 1e-2, tr))

        raise ValueError("max tries ({}) exceeded".format(str(1e6)))

    return _fn


class BestFirstSearch(TracePosterior):
    """
    Inference by enumerating executions ordered by their probabilities.
    Exact (and results equivalent to Search) if all executions are enumerated.
    """
    def __init__(self, model, num_samples=None, **kwargs):
        if num_samples is None:
            num_samples = 100
        self.num_samples = num_samples
        self.model = model
        super(BestFirstSearch, self).__init__(**kwargs)

    def _traces(self, *args, **kwargs):
        q = queue.PriorityQueue()
        # add a little bit of noise to the priority to break ties...
        q.put((torch.zeros(1).item() - torch.rand(1).item() * 1e-2, poutine.Trace()))
        q_fn = pqueue(self.model, queue=q)
        for i in range(self.num_samples):
            if q.empty():
                # num_samples was too large!
                break
            tr = poutine.trace(q_fn).get_trace(*args, **kwargs)  # XXX should block
            yield tr, tr.log_prob_sum()
