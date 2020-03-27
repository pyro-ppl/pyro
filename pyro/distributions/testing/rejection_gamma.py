# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.distributions.rejector import Rejector
from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.torch import Beta, Dirichlet, Gamma, Normal
from pyro.distributions.util import copy_docs_from, weakmethod


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
        self._standard_gamma = Gamma(concentration, concentration.new([1.]).squeeze().expand_as(concentration))
        # The following are Marsaglia & Tsang's variable names.
        self._d = self.concentration - 1.0 / 3.0
        self._c = 1.0 / torch.sqrt(9.0 * self._d)
        # Compute log scale using Gamma.log_prob().
        x = self._d.detach()  # just an arbitrary x.
        log_scale = self.propose_log_prob(x) + self.log_prob_accept(x) - self.log_prob(x)
        super().__init__(self.propose, self.log_prob_accept, log_scale,
                         batch_shape=concentration.shape, event_shape=())

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RejectionStandardGamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new._standard_gamma = self._standard_gamma.expand(batch_shape)
        new._d = self._d.expand(batch_shape)
        new._c = self._c.expand(batch_shape)
        # Compute log scale using Gamma.log_prob().
        x = new._d.detach()  # just an arbitrary x.
        log_scale = new.propose_log_prob(x) + new.log_prob_accept(x) - new.log_prob(x)
        super(RejectionStandardGamma, new).__init__(new.propose, new.log_prob_accept, log_scale,
                                                    batch_shape=batch_shape, event_shape=())
        new._validate_args = self._validate_args
        return new

    @weakmethod
    def propose(self, sample_shape=torch.Size()):
        # Marsaglia & Tsang's x == Naesseth's epsilon`
        x = torch.randn(sample_shape + self.concentration.shape,
                        dtype=self.concentration.dtype,
                        device=self.concentration.device)
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

    @weakmethod
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
        super().__init__(concentration, rate, validate_args=validate_args)
        self._standard_gamma = RejectionStandardGamma(concentration)
        self.rate = rate

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RejectionGamma, _instance)
        new = super().expand(batch_shape, new)
        new._standard_gamma = self._standard_gamma.expand(batch_shape)
        new._validate_args = self._validate_args
        return new

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
        super().__init__(concentration, rate, validate_args=validate_args)
        self.concentration = concentration
        self._boost = boost
        self._rejection_gamma = RejectionGamma(concentration + boost, rate)
        self._unboost_x_cache = None, None

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ShapeAugmentedGamma, _instance)
        new = super().expand(batch_shape, new)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new._boost = self._boost
        new._rejection_gamma = self._rejection_gamma.expand(batch_shape)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        x = self._rejection_gamma.rsample(sample_shape)
        boosted_x = x.clone()
        for i in range(self._boost):
            u = torch.rand(x.shape, dtype=x.dtype, device=x.device)
            boosted_x *= (1 - u) ** (1 / (i + self.concentration))
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
        super().__init__(concentration, validate_args=validate_args)
        self._gamma = ShapeAugmentedGamma(concentration, torch.ones_like(concentration), boost)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ShapeAugmentedDirichlet, _instance)
        new = super().expand(batch_shape, new)
        batch_shape = torch.Size(batch_shape)
        new._gamma = self._gamma.expand(batch_shape + self._gamma.concentration.shape[-1:])
        new._validate_args = self._validate_args
        return new

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
        super().__init__(concentration1, concentration0, validate_args=validate_args)
        alpha_beta = torch.stack([concentration1, concentration0], -1)
        self._gamma = ShapeAugmentedGamma(alpha_beta, torch.ones_like(alpha_beta), boost)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ShapeAugmentedBeta, _instance)
        new = super().expand(batch_shape, new)
        batch_shape = torch.Size(batch_shape)
        new._gamma = self._gamma.expand(batch_shape + self._gamma.concentration.shape[-1:])
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        gammas = self._gamma.rsample(sample_shape)
        probs = gammas / gammas.sum(-1, True)
        return probs[..., 0]
