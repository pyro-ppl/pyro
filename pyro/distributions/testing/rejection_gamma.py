from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.rejector import Rejector
from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.torch import Beta, Dirichlet, Gamma, Normal
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Gamma)
class RejectionStandardGamma(Rejector):
    """
    Naive Marsaglia & Tsang rejection sampler for standard Gamma distibution.
    This assumes `concentration >= 1` and does not boost `concentration` or augment shape.
    """
    def __init__(self, concentration):
        if concentration.data.min() < 1:
            raise NotImplementedError('concentration < 1 is not supported')
        self.concentration = concentration
        self._standard_gamma = Gamma(concentration, concentration.new_tensor([1.]).squeeze().expand_as(concentration))
        # The following are Marsaglia & Tsang's variable names.
        self._d = self.concentration - 1.0 / 3.0
        self._c = 1.0 / torch.sqrt(9.0 * self._d)
        # Compute log scale using Gamma.log_prob().
        x = self._d.detach()  # just an arbitrary x.
        log_scale = self.propose_log_prob(x) + self.log_prob_accept(x) - self.log_prob(x)
        super(RejectionStandardGamma, self).__init__(self.propose, self.log_prob_accept, log_scale)

    def propose(self, sample_shape=torch.Size()):
        # Marsaglia & Tsang's x == Naesseth's epsilon
        x = self.concentration.new_empty(sample_shape + self.concentration.shape).normal_()
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
        result += Normal(torch.zeros_like(self.concentration), torch.ones_like(self.concentration)).log_prob(x)
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
    has_rsample = True

    def __init__(self, concentration, rate, validate_args=None):
        super(RejectionGamma, self).__init__(concentration, rate, validate_args=validate_args)
        self._standard_gamma = RejectionStandardGamma(concentration)
        self.rate = rate

    def rsample(self, sample_shape=torch.Size()):
        return self._standard_gamma.rsample(sample_shape) / self.rate

    def log_prob(self, x):
        return self._standard_gamma.log_prob(x * self.rate) + torch.log(self.rate)

    def score_parts(self, x):
        log_prob, score_function, _ = self._standard_gamma.score_parts(x * self.rate)
        log_prob = log_prob + torch.log(self.rate)
        return ScoreParts(log_prob, score_function, log_prob)


@copy_docs_from(Gamma)
class ShapeAugmentedGamma(Gamma):
    """
    This implements the shape augmentation trick of
    Naesseth, Ruiz, Linderman, Blei (2017) https://arxiv.org/abs/1610.05683
    """
    has_rsample = True

    def __init__(self, concentration, rate, boost=1, validate_args=None):
        if concentration.min() + boost < 1:
            raise ValueError('Need to boost at least once for concentration < 1')
        super(ShapeAugmentedGamma, self).__init__(concentration, rate, validate_args=validate_args)
        self.concentration = concentration
        self._boost = boost
        self._rejection_gamma = RejectionGamma(concentration + boost, rate)
        self._unboost_x_cache = None, None

    def rsample(self, sample_shape=torch.Size()):
        x = self._rejection_gamma.rsample(sample_shape)
        boosted_x = x.clone()
        for i in range(self._boost):
            boosted_x *= (1 - x.new_empty(x.shape).uniform_()) ** (1 / (i + self.concentration))
        self._unboost_x_cache = boosted_x, x
        return boosted_x

    def score_parts(self, boosted_x=None):
        if boosted_x is None:
            boosted_x = self._unboost_x_cache[0]
        assert boosted_x is self._unboost_x_cache[0]
        x = self._unboost_x_cache[1]
        _, score_function, _ = self._rejection_gamma.score_parts(x)
        log_prob = self.log_prob(boosted_x)
        return ScoreParts(log_prob, score_function, log_prob)


@copy_docs_from(Dirichlet)
class ShapeAugmentedDirichlet(Dirichlet):
    """
    Implementation of ``Dirichlet`` via ``ShapeAugmentedGamma``.

    This naive implementation has stochastic reparameterized gradients, which
    have higher variance than PyTorch's ``Dirichlet`` implementation.
    """
    def __init__(self, concentration, boost=1, validate_args=None):
        super(ShapeAugmentedDirichlet, self).__init__(concentration, validate_args=validate_args)
        self._gamma = ShapeAugmentedGamma(concentration, torch.ones_like(concentration), boost)

    def rsample(self, sample_shape=torch.Size()):
        gammas = self._gamma.rsample(sample_shape)
        return gammas / gammas.sum(-1, True)


@copy_docs_from(Beta)
class ShapeAugmentedBeta(Beta):
    """
    Implementation of ``rate`` via ``ShapeAugmentedGamma``.

    This naive implementation has stochastic reparameterized gradients, which
    have higher variance than PyTorch's ``rate`` implementation.
    """
    def __init__(self, concentration1, concentration0, boost=1, validate_args=None):
        super(ShapeAugmentedBeta, self).__init__(concentration1, concentration0, validate_args=validate_args)
        alpha_beta = torch.stack([concentration1, concentration0], -1)
        self._gamma = ShapeAugmentedGamma(alpha_beta, torch.ones_like(alpha_beta), boost)

    def rsample(self, sample_shape=torch.Size()):
        gammas = self._gamma.rsample(sample_shape)
        probs = gammas / gammas.sum(-1, True)
        return probs[..., 0]
