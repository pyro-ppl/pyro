import torch
import numpy as np
import pyro.distributions
import pyro.util
import pyro.poutine

from pyro.infer.abstract_infer import AbstractInfer
from pyro.infer.search import Search
from pyro.infer.mh import MH
from pyro.infer.importance import Importance
from pyro.infer.kl_qp import KL_QP


class Marginal(pyro.distributions.Distribution):
    """
    Marginal histogram
    """
    def __init__(self, trace_dist):
        assert isinstance(trace_dist, AbstractInfer), \
            "trace_dist must be trace posterior distribution object"
        super(Marginal, self).__init__()
        self.trace_dist = trace_dist

    @pyro.util.memoize
    def _dist(self, *args, **kwargs):
        """
        Convert a histogram over traces to a histogram over return values
        Currently very inefficient...
        """
        vs, log_weights = [], []
        for tr, log_weight in self.trace_dist._traces(*args, **kwargs):
            vs.append(tr["_RETURN"]["value"])
            log_weights.append(log_weight)

        log_weights = torch.cat(log_weights)
        if not isinstance(log_weights, torch.autograd.Variable):
            log_weights = torch.autograd.Variable(log_weights)
        log_z = pyro.util.log_sum_exp(log_weights)
        ps = torch.exp(log_weights - log_z.expand_as(log_weights))

        if isinstance(vs[0], (torch.autograd.Variable, torch.Tensor, np.ndarray)):
            hist = pyro.util.tensor_histogram(ps, vs)
        else:
            hist = pyro.util.basic_histogram(ps, vs)
        return pyro.distributions.Categorical(ps=hist["ps"], vs=hist["vs"])

    def sample(self, *args, **kwargs):
        return pyro.poutine.block(self._dist(*args, **kwargs)).sample()

    def log_pdf(self, val, *args, **kwargs):
        return pyro.poutine.block(self._dist(*args, **kwargs)).log_pdf(val)

    def support(self, *args, **kwargs):
        return pyro.poutine.block(self._dist(*args, **kwargs)).support()
