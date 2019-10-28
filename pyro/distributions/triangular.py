from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from pyro.distributions import TorchDistribution, Uniform


class Triangular(TorchDistribution):
    """
     A Triangular distribution.

    :param torch.Tensor low: lower range (inclusive).
    :param torch.Tensor high: upper range (inclusive).
    :param torch.Tensor peak: mode of range.
    """
    arg_constraints = {'low': constraints.real, 'high': constraints.dependent, 'peak': constraints.dependent}
    support = constraints.dependent
    has_rsample = True

    @property
    def mean(self):
        return (self.low + self.high + self.peak) / 3

    @property
    def variance(self):
        return (self.low**2 + self.high**2 + self.peak**2 - self.low * self.high - self.low * self.peak -
                self.high * self.peak) / 18

    def __init__(self, low, high, peak, validate_args=None):
        self.low, self.high, self.peak = broadcast_all(low, high, peak)
        self._uniform = Uniform(0, 1)

        if isinstance(low, Number) and isinstance(high, Number) and isinstance(peak, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.shape
        event_shape = torch.Size()
        super(Triangular, self).__init__(batch_shape, event_shape, validate_args)

        if self._validate_args:
            if not torch.lt(self.low, self.high).all():
                raise ValueError("Triangle is not defined when low >= high")
            if not torch.le(self.low, self.peak).all():
                raise ValueError("Triangle is not defined when peak < low")
            if not torch.le(self.peak, self.high).all():
                raise ValueError("Triangle is not defined when peak > high")

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        interval_length = self.high - self.low
        samples = self._uniform.rsample(shape)
        return torch.where(samples < (self.peak - self.low) / interval_length,
                           self.low + torch.sqrt(samples * interval_length * (self.peak - self.low)),
                           self.high - torch.sqrt((1. - samples) * interval_length * (self.high - self.peak)))

    def log_prob(self, value):
        interval_length = self.high - self.low
        result_inside_interval = torch.where((value >= self.low) & (value <= self.peak),
                                             torch.tensor(2.0).log() + torch.log(value - self.low) -
                                             torch.log(interval_length) - torch.log(self.peak - self.low),
                                             torch.tensor(2.0).log() + torch.log(self.high - value) -
                                             torch.log(interval_length) - torch.log(self.high - self.peak))
        return torch.where((value < self.low) | (value > self.high), torch.log(torch.ones_like(value) * 1e-6),
                           result_inside_interval)

    def expand(self, batch_shape):
        try:
            return super(Triangular, self).expand(batch_shape)
        except NotImplementedError:
            low = self.low.expand(batch_shape)
            high = self.high.expand(batch_shape)
            peak = self.peak.expand(batch_shape)
            result = type(self)(low, high, peak, validate_args=False)
            result._validate_args = self.__dict__.get('_validate_args')
            return result
