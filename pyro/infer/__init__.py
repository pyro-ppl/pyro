import torch
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
    def _aggregate(self, trace_hist):
        """
        Convert a histogram over traces to a histogram over return values
        Currently very inefficient...
        """
        assert isinstance(trace_hist, pyro.distributions.Categorical), \
            "trace histogram must be a Categorical distribution object"
        hist = dict()
        for i, tr in enumerate(trace_hist.vs[0]):
            v = tr["_RETURN"]["value"]
            if v not in hist:
                hist[v] = 0.0
            hist[v] = hist[v] + trace_hist.ps[0][i]
        return pyro.distributions.Categorical(ps=torch.cat([vv for vv in hist.values()]),
                                              vs=[[kk for kk in hist.keys()]])

    def sample(self, *args, **kwargs):
        return self._aggregate(
            pyro.poutine.block(self.trace_dist._dist)(*args, **kwargs)).sample()

    def log_pdf(self, val, *args, **kwargs):
        return self._aggregate(
            pyro.poutine.block(self.trace_dist._dist)(*args, **kwargs)).log_pdf(val)

    def support(self, *args, **kwargs):
        return self._aggregate(
            pyro.poutine.block(self.trace_dist._dist)(*args, **kwargs)).support()
