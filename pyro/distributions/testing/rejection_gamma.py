from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.rejector import Rejector
from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.torch.gamma import Gamma
from pyro.distributions.torch.normal import Normal
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Gamma)
class RejectionStandardGamma(Rejector):
    """
    Naive Marsaglia & Tsang rejection sampler for standard Gamma distibution.
    This assumes `alpha >= 1` and does not boost `alpha` or augment shape.
    """
    def __init__(self, alpha):
        if alpha.data.min() < 1:
            raise NotImplementedError('alpha < 1 is not supported')
        self.alpha = alpha
        self._standard_gamma = Gamma(alpha, alpha.new([1]).expand_as(alpha))
        # The following are Marsaglia & Tsang's variable names.
        self._d = self.alpha - 1.0 / 3.0
        self._c = 1.0 / torch.sqrt(9.0 * self._d)
        # Compute log scale using Gamma.log_prob().
        x = self._d.detach()  # just an arbitrary x.
        log_scale = self.propose_log_prob(x) + self.log_prob_accept(x) - self.log_prob(x)
        super(RejectionStandardGamma, self).__init__(self.propose, self.log_prob_accept, log_scale)

    def propose(self):
        # Marsaglia & Tsang's x == Naesseth's epsilon
        x = self.alpha.new(self.alpha.shape).normal_()
        y = 1.0 + self._c * x
        v = y * y * y
        return (self._d * v).clamp_(1e-30, 1e30)

    def propose_log_prob(self, value):
        v = value / self._d
        result = -self._d.log()
        y = v.pow(1 / 3)
        result -= torch.log(3 * y ** 2)
        x = (y - 1) / self._c
        result -= self._c.log()
        result += Normal(torch.zeros_like(self.alpha), torch.ones_like(self.alpha)).log_prob(x)
        return result

    def log_prob_accept(self, value):
        v = value / self._d
        y = torch.pow(v, 1.0 / 3.0)
        x = (y - 1.0) / self._c
        log_prob_accept = 0.5 * x * x + self._d * (1.0 - v + torch.log(v))
        log_prob_accept[y <= 0] = -float('inf')
        return log_prob_accept

    def log_prob(self, x):
        return self._standard_gamma.log_prob(x)


@copy_docs_from(Gamma)
class RejectionGamma(Gamma):
    stateful = True
    reparameterized = True

    def __init__(self, alpha, beta):
        super(RejectionGamma, self).__init__(alpha, beta)
        self._standard_gamma = RejectionStandardGamma(alpha)
        self.beta = beta

    def sample(self):
        return self._standard_gamma.sample() / self.beta

    def log_prob(self, x):
        return self._standard_gamma.log_prob(x * self.beta) + torch.log(self.beta)

    def score_parts(self, x):
        log_pdf, score_function, _ = self._standard_gamma.score_parts(x * self.beta)
        log_pdf = log_pdf + torch.log(self.beta)
        return ScoreParts(log_pdf, score_function, log_pdf)


@copy_docs_from(Gamma)
class ShapeAugmentedGamma(Gamma):
    """
    This implements the shape augmentation trick of
    Naesseth, Ruiz, Linderman, Blei (2017) https://arxiv.org/abs/1610.05683
    """
    stateful = True
    reparameterized = True

    def __init__(self, alpha, beta, boost=1):
        if alpha.min() + boost < 1:
            raise ValueError('Need to boost at least once for alpha < 1')
        super(ShapeAugmentedGamma, self).__init__(alpha, beta)
        self.alpha = alpha
        self._boost = boost
        self._rejection_gamma = RejectionGamma(alpha + boost, beta)
        self._unboost_x_cache = None, None

    def sample(self):
        x = self._rejection_gamma.sample()
        boosted_x = x.clone()
        for i in range(self._boost):
            boosted_x *= (1 - x.new(x.shape).uniform_()) ** (1 / (i + self.alpha))
        self._unboost_x_cache = boosted_x, x
        return boosted_x

    def score_parts(self, boosted_x=None):
        if boosted_x is None:
            boosted_x = self._unboost_x_cache[0]
        assert boosted_x is self._unboost_x_cache[0]
        x = self._unboost_x_cache[1]
        _, score_function, _ = self._rejection_gamma.score_parts(x)
        log_pdf = self.log_prob(boosted_x)
        return ScoreParts(log_pdf, score_function, log_pdf)
