import numpy as np
import torch
from torch.autograd import Variable
import pyro
import pyro.util
from pyro.distributions import Categorical

import pdb


class Histogram(pyro.distributions.Distribution):
    """
    Histogram
    """
    @pyro.util.memoize
    def _dist(self, *args, **kwargs):
        """
        Convert a histogram over traces to a histogram over return values
        Currently very inefficient...
        """
        vs, log_weights = [], []
        for v, log_weight in self._gen_weighted_samples(*args, **kwargs):
            vs.append(v)
            log_weights.append(log_weight)

        log_weights = torch.cat(log_weights)
        if not isinstance(log_weights, torch.autograd.Variable):
            log_weights = torch.autograd.Variable(log_weights)
        log_z = pyro.util.log_sum_exp(log_weights)
        ps = torch.exp(log_weights - log_z.expand_as(log_weights))

        # pdb.set_trace()
        if isinstance(vs[0], (torch.autograd.Variable, torch.Tensor, np.ndarray)):
            hist = pyro.util.tensor_histogram(ps, vs)
        else:
            hist = pyro.util.basic_histogram(ps, vs)
        return pyro.distributions.Categorical(ps=hist["ps"], vs=hist["vs"])

    def _gen_weighted_samples(self, *args, **kwargs):
        raise NotImplementedError("_gen_weighted_samples is abstract method")

    def sample(self, *args, **kwargs):
        return pyro.poutine.block(self._dist)(*args, **kwargs).sample()

    def log_pdf(self, val, *args, **kwargs):
        return pyro.poutine.block(self._dist)(*args, **kwargs).log_pdf(val)

    def support(self, *args, **kwargs):
        return pyro.poutine.block(self._dist)(*args, **kwargs).support()


class Marginal(Histogram):
    """
    Marginal histogram
    """
    def __init__(self, trace_dist):
        assert isinstance(trace_dist, TracePosterior), \
            "trace_dist must be trace posterior distribution object"
        super(Marginal, self).__init__()
        self.trace_dist = trace_dist

    def _gen_weighted_samples(self, *args, **kwargs):
        for tr, log_weight in self.trace_dist._traces(*args, **kwargs):
            yield (tr["_RETURN"]["value"], log_weight)


class TracePosterior(object):
    """
    abstract inference class
    TODO documentation
    """
    def __init__(self):
        pass
 
    def _gen_weighted_samples(self, *args, **kwargs):
        for tr, log_weight in self._traces(*args, **kwargs):
            yield (tr, log_weight)

    def _traces(self, *args, **kwargs):
        """
        Virtual method to get unnormalized weighted list of posterior traces
        """
        raise NotImplementedError("inference algorithm must implement _traces")

    # XXX this isnt a good abstraction for marginal likelihood estimation
    def log_z(self, *args, **kwargs):
        """
        estimate marginal probability of observations
        """
        raise NotImplementedError("inference algorithm must implement log_z")
