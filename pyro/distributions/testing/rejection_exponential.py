from __future__ import absolute_import, division, print_function

import torch
from torch.distributions.utils import broadcast_all

from pyro.distributions.rejector import Rejector
from pyro.distributions.torch import Exponential
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Exponential)
class RejectionExponential(Rejector):
    def __init__(self, rate, factor):
        assert (factor <= 1).all()
        self.rate, self.factor = broadcast_all(rate, factor)
        propose = Exponential(self.factor * self.rate)
        log_scale = self.factor.log()
        super(RejectionExponential, self).__init__(propose, self.log_prob_accept, log_scale)

    def log_prob_accept(self, x):
        result = (self.factor - 1) * self.rate * x
        assert result.max() <= 0, result.max()
        return result

    @property
    def batch_shape(self):
        return self.rate.shape

    @property
    def event_shape(self):
        return torch.Size()
