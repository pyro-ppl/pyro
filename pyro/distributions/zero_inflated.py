import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property

from pyro.distributions import TorchDistribution, Poisson, NegativeBinomial


class ZeroInflatedDistribution(TorchDistribution):
    """
    Base class for a Zero Inflated distribution.

    :param torch.Tensor gate: probability of extra zeros given via a Bernoulli distribution.
    :param TorchDistribution base_dist: the base distribution.
    """
    arg_constraints = {"gate": constraints.unit_interval}

    def __init__(self, gate, base_dist, validate_args=None):
        self.gate = gate
        self.base_dist = base_dist
        batch_shape = self.gate.shape
        event_shape = torch.Size()

        super(ZeroInflatedDistribution, self).__init__(
            batch_shape, event_shape, validate_args
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        gate, value = broadcast_all(self.gate, value)
        log_prob = (-gate).log1p() + self.base_dist.log_prob(value)
        log_prob = torch.where(value == 0, (gate + log_prob.exp()).log(), log_prob)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * (
            self.base_dist.mean ** 2 + self.base_dist.variance
        ) - (self.mean) ** 2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        gate = self.gate.expand(batch_shape)
        base_dist = self.base_dist.expand(batch_shape)
        ZeroInflatedDistribution.__init__(new, gate, base_dist, validate_args=False)
        new._validate_args = self._validate_args
        return new


class ZeroInflatedPoisson(ZeroInflatedDistribution):
    """
    A Zero Inflated Poisson distribution.

    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor rate: rate of poisson distribution.
    """
    support = constraints.nonnegative_integer

    def __init__(self, gate, rate, validate_args=None):
        base_dist = Poisson(rate=rate, validate_args=validate_args)

        super(ZeroInflatedPoisson, self).__init__(
            gate, base_dist, validate_args=validate_args
        )


class ZeroInflatedNegativeBinomial(ZeroInflatedDistribution):
    """
    A Zero Inflated Negative Binomial distribution.

    :param torch.Tensor gate: probability of extra zeros.
    :param total_count (float or Tensor): non-negative number of negative Bernoulli trials
    :param probs (Tensor): Event probabilities of success in the half open interval [0, 1)
    :param logits (Tensor): Event log-odds for probabilities of success
    """
    support = constraints.nonnegative_integer

    def __init__(self, gate, total_count, probs=None, logits=None, validate_args=None):
        base_dist = NegativeBinomial(
            total_count=total_count,
            probs=probs,
            logits=logits,
            validate_args=validate_args,
        )

        super(ZeroInflatedNegativeBinomial, self).__init__(
            gate, base_dist, validate_args=validate_args
        )
