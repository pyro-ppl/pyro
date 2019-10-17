import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property

from pyro.distributions import TorchDistribution


class ZeroInflatedPoisson(TorchDistribution):
    """
    A Zero Inflated Poisson distribution.

    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor rate: rate of poisson distribution.
    """
    arg_constraints = {'gate': constraints.unit_interval, 'rate': constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, gate, rate, validate_args=None):
        self.gate, self.rate = broadcast_all(gate, rate)
        batch_shape = self.gate.shape
        event_shape = torch.Size()
        super(ZeroInflatedPoisson, self).__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        gate, rate, value = broadcast_all(self.gate, self.rate, value)
        log_prob = (-gate).log1p() + (rate.log() * value) - rate - (value + 1).lgamma()
        log_prob = torch.where(value == 0, (gate + log_prob.exp()).log(), log_prob)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = torch.poisson(self.rate.expand(shape))
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.rate

    @lazy_property
    def variance(self):
        return self.rate * (1 - self.gate) * (1 + self.rate * self.gate)

    def expand(self, batch_shape):
        try:
            return super(ZeroInflatedPoisson, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            gate = self.gate.expand(batch_shape)
            rate = self.rate.expand(batch_shape)
            return type(self)(gate, rate, validate_args=validate_args)
