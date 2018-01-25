from __future__ import absolute_import, division, print_function

from pyro.distributions import Exponential
from pyro.distributions.rejector import Rejector
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Exponential)
class RejectionExponential(Rejector):
    def __init__(self, rate, factor):
        assert (factor <= 1).all()
        self.rate = rate
        self.factor = factor
        propose = Exponential(self.factor * self.rate)
        log_scale = self.factor.log()
        super(RejectionExponential, self).__init__(propose, self.log_prob_accept, log_scale)

    def log_prob_accept(self, x):
        result = (self.factor - 1) * self.rate * x
        assert result.max() <= 0, result.max()
        return result
