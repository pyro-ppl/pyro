import numpy as np
import torch
from torch.autograd import Variable

import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.util as util


class Histogram(dist.Distribution):
    """
    Abstract Histogram distribution.  For now, should not be using outside Marginal.
    """
    enumerable = True

    @util.memoize
    def _dist(self, *args, **kwargs):
        """
        This is an abstract method
        """
        # XXX currently this whole object is very inefficient
        vs, log_weights = [], []
        for v, log_weight in self._gen_weighted_samples(*args, **kwargs):
            vs.append(v)
            log_weights.append(log_weight)

        log_weights = torch.cat(log_weights)
        if not isinstance(log_weights, torch.autograd.Variable):
            log_weights = Variable(log_weights)
        log_z = util.log_sum_exp(log_weights)
        ps = torch.exp(log_weights - log_z.expand_as(log_weights))

        if isinstance(vs[0], (Variable, torch.Tensor, np.ndarray)):
            hist = util.tensor_histogram(ps, vs)
        else:
            hist = util.basic_histogram(ps, vs)
        return dist.Categorical(ps=hist["ps"], vs=hist["vs"])

    def _gen_weighted_samples(self, *args, **kwargs):
        raise NotImplementedError("_gen_weighted_samples is abstract method")

    def sample(self, *args, **kwargs):
        return poutine.block(self._dist)(*args, **kwargs).sample()[0]

    def log_pdf(self, val, *args, **kwargs):
        return poutine.block(self._dist)(*args, **kwargs).log_pdf([val])

    def batch_log_pdf(self, val, *args, **kwargs):
        return poutine.block(self._dist)(*args, **kwargs).batch_log_pdf([val])

    def support(self, *args, **kwargs):
        return poutine.block(self._dist)(*args, **kwargs).support()


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
        traces = []
        log_weights = []
        for tr, log_w in poutine.block(self._traces)(*args, **kwargs):
            traces.append(tr)
            log_weights.append(log_w)
        log_weights = torch.cat(log_weights)
        if not isinstance(log_weights, torch.autograd.Variable):
            log_weights = Variable(log_weights)
        log_z = util.log_sum_exp(log_weights)
        ps = torch.exp(log_weights - log_z.expand_as(log_weights))
        ix = dist.categorical(ps=ps, one_hot=False)
        return traces[ix.data[0]]
