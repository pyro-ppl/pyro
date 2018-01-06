from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.distribution import Distribution
from pyro.distributions.gamma import Gamma
from pyro.distributions.rejector import ImplicitRejector
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Gamma)
class RejectionStandardGamma(ImplicitRejector):
    """
    Naive Marsaglia & Tsang rejection sampler for standard Gamma distibution.
    This assumes `alpha >= 1` and does not boost `alpha` boosting or
    augment shape.
    """
    def __init__(self, alpha):
        super(RejectionStandardGamma, self).__init__(self.propose, self.log_prob_accept)
        if not (alpha >= 1).all():
            raise NotImplementedError('alpha < 1 is not supported')
        self.alpha = alpha
        # The following are Marsaglia & Tsang's variable names.
        self._d = self.alpha - 1.0 / 3.0
        self._c = 1.0 / torch.sqrt(9.0 * self._d)

    def propose(self):
        # Marsaglia & Tsang's x == Naesseth's epsilon
        x = self.alpha.new(self.alpha.shape).normal_()
        y = 1.0 + self._c * x
        v = y * y * y
        return (self._d * v).clamp_(1e-30, 1e30)

    def log_prob_accept(self, value):
        v = value / self._d
        y = torch.pow(v, 1.0 / 3.0)
        x = (y - 1.0) / self._c
        log_prob_accept = 0.5 * x * x + self._d * (1.0 - v + torch.log(v))
        log_prob_accept[y <= 0] = -float('inf')
        return log_prob_accept


"""
It's now easy to implement a full Gamma distribution on top of
our `RejectionStandardGamma`:
"""


@copy_docs_from(Gamma)
class RejectionGamma(Distribution):
    reparameterized = True

    def __init__(self, alpha, beta):
        self._standard_gamma = RejectionStandardGamma(alpha)
        self.beta = beta

    def sample(self):
        return self._standard_gamma.sample() * self.beta

    def batch_log_pdf(self, x):
        return self._standard_gamma.batch_log_pdf(x / self.beta) - torch.log(self.beta)

    def score_function_term(self, x):
        return self._standard_gamma(x / self.beta)
