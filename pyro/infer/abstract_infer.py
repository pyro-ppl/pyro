import numpy as np
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.util


class Histogram(pyro.distributions.Distribution):
    """
    Abstract Histogram distribution.  For now, should not be using outside Marginal.
    """
    enumerable = True

    @pyro.util.memoize
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
        log_z = pyro.util.log_sum_exp(log_weights)
        ps = torch.exp(log_weights - log_z.expand_as(log_weights))

        if isinstance(vs[0], (Variable, torch.Tensor, np.ndarray)):
            hist = pyro.util.tensor_histogram(ps, vs)
        else:
            hist = pyro.util.basic_histogram(ps, vs)
        return dist.Categorical(ps=hist["ps"], vs=hist["vs"])

    def _gen_weighted_samples(self, *args, **kwargs):
        raise NotImplementedError("_gen_weighted_samples is abstract method")

    def sample(self, *args, **kwargs):
        return poutine.block(self._dist)(*args, **kwargs).sample()[0]

    def log_pdf(self, val, *args, **kwargs):
        return poutine.block(self._dist)(*args, **kwargs).log_pdf([val])

    def support(self, *args, **kwargs):
        return poutine.block(self._dist)(*args, **kwargs).support()


class Marginal(Histogram):
    """
    :param trace_dist: a TracePosterior instance representing a Monte Carlo posterior

    Marginal histogram distribution.
    Turns a TracePosterior object into a Distribution
    over the return values of the TracePosterior's model.
    """
    def __init__(self, trace_dist):
        assert isinstance(trace_dist, TracePosterior), \
            "trace_dist must be trace posterior distribution object"
        super(Marginal, self).__init__()
        self.trace_dist = trace_dist

    def _gen_weighted_samples(self, *args, **kwargs):
        for tr, log_weight in self.trace_dist._traces(*args, **kwargs):
            yield (tr.nodes["_RETURN"]["value"], log_weight)


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

    def log_z(self, *args, **kwargs):
        """
        Abstract method.
        Algorithm-specific estimate of marginal log-probability of observations.
        Should have same input signature as self.model and self.guide and return a scalar.
        """
        raise NotImplementedError("inference algorithm must implement log_z")
