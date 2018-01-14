from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.distribution import Distribution
from pyro.distributions.gamma import Gamma
from pyro.distributions.rejector import ImplicitRejector
from pyro.distributions.score_parts import ScoreParts
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
        if alpha.data.min() < 1:
            raise NotImplementedError('alpha < 1 is not supported')
        self.alpha = alpha
        self._standard_gamma = Gamma(alpha, alpha.new([1]).expand_as(alpha))
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

    def batch_log_pdf(self, x):
        return self._standard_gamma.batch_log_pdf(x)


# Note it's easy to implement a full Gamma distribution on top of
# our `StandardGamma`:

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

    def score_parts(self, x):
        log_pdf, score_function, _ = self._standard_gamma.score_parts(x / self.beta)
        log_pdf = log_pdf - torch.log(self.beta)
        return ScoreParts(log_pdf, score_function, log_pdf)


# Next let's implement Shape Augmentation.

@copy_docs_from(Gamma)
class ShapeAugmentedGamma(Gamma):
    reparameterized = True

    def __init__(self, alpha, beta, boost=1):
        if alpha.min() + boost < 1:
            raise ValueError('Need to boost at least once for alpha < 1')
        super(ShapeAugmentedGamma, self).__init__(alpha, beta)
        self._boost = boost
        self._rejection_gamma = RejectionStandardGamma(alpha + boost)
        self._unboost_x_cache = None, None

    def sample(self):
        x = self._rejection_gamma.sample()
        boosted_x = x.clone()
        for i in range(self._boost):
            boosted_x *= (1 - x.new(x.shape).uniform_()) ** (1 / (i + self.alpha))
        self._unboost_x_cache = boosted_x / self.beta, x
        return self._unboost_x_cache[0]

    def score_parts(self, boosted_x):
        assert self._unboost_x_cache[0] is boosted_x
        x = self._unboost_x_cache[1]
        score_function = self._rejection_gamma.score_parts(x)[1]
        log_pdf = self.batch_log_pdf(boosted_x)
        return ScoreParts(log_pdf, score_function, log_pdf)


def kl_gamma_gamma(p, q):
    p = getattr(p, 'torch_dist', p)
    q = getattr(q, 'torch_dist', q)

    # Adapted from https://stats.stackexchange.com/questions/11646
    def f(a, b, c, d):
        return -d * a / c + b * a.log() - torch.lgamma(b) + (b - 1) * torch.digamma(d) + (1 - b) * c.log()

    return f(p.beta, p.alpha, p.beta, p.alpha) - f(q.beta, q.alpha, p.beta, p.alpha)
