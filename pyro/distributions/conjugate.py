from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from pyro.distributions.torch import Beta, Binomial, Gamma, Poisson
from pyro.distributions.torch_distribution import TorchDistribution


def _log_beta(x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)


class BetaBinomial(TorchDistribution):
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive,
                       'total_count': constraints.nonnegative_integer}
    support = Binomial.support

    def __init__(self, concentration1, concentration0, total_count=1, validate_args=None):
        concentration1, concentration0, total_count = broadcast_all(
            concentration1, concentration0, total_count)
        self._beta = Beta(concentration1, concentration0)
        self.total_count = total_count
        super(BetaBinomial, self).__init__(self._beta._batch_shape, validate_args=validate_args)

    @property
    def concentration1(self):
        return self._beta.concentration1

    @property
    def concentration0(self):
        return self._beta.concentration0

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(BetaBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new._beta = self._beta.expand(batch_shape)
        new.total_count = self.total_count.expand_as(new._beta.concentration0)
        super(BetaBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=()):
        probs = self.Beta.sample(sample_shape)
        return Binomial(probs).sample()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        return (log_factorial_n - log_factorial_k - log_factorial_nmk +
                _log_beta(value + self.concentration1, self.total_count - value + self.concentration0) -
                _log_beta(self.concentration0, self.concentration1))

    @property
    def mean(self):
        return self._beta.mean * self.total_count

    @property
    def variance(self):
        return self._beta.variance * self.total_count * (self.concetration0 + self.concentration1 + self.total_count)


class GammaPoisson(TorchDistribution):
    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = Poisson.support

    def __init__(self, concentration, rate, validate_args=None):
        concentration, rate = broadcast_all(concentration, rate)
        self._gamma = Gamma(concentration, rate)
        super(GammaPoisson, self).__init__(self._gamma._batch_shape, validate_args=validate_args)

    @property
    def concentration(self):
        return self._gamma.concentration

    @property
    def rate(self):
        return self._gamma.rate

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GammaPoisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new._beta = self._gamma.expand(batch_shape)
        super(GammaPoisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=()):
        rate = self.Gamma.sample(sample_shape)
        return Poisson(rate).sample()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.lgamma(self.rate + value) + value * self.concentration.log() \
            - torch.lgamma(self.rate) - (self.rate + value) * (1 + self.concentration).log() \
            - torch.lgamma(value + 1)

    @property
    def mean(self):
        return self.concentration * self.rate

    @property
    def variance(self):
        return self.concentration * self.rate * (1 + self.concentration)
