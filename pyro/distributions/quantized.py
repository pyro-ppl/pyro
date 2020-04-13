# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions.utils import broadcast_all, lazy_property
from torch.distributions import constraints

from .conjugate import BetaBinomial
from .torch import Beta, Binomial
from .torch_distribution import TorchDistribution


def quantize(rate):
    """
    Quantize a real-valued tensor by randomly rounding to the nearest two
    integer neighbors, based on distance to each of those neighbors.
    """
    with torch.no_grad():
        lb = rate.unsqueeze(-1).floor()
        ub = lb + 1
        counts = torch.cat([lb, ub], dim=-1)
    probs = rate - lb
    return probs, counts


@torch.no_grad()
def dequantize(count, int_dist):
    """
    Given a sample ``count`` from an nonnegative integer valued distribution
    ``int_dist``, add dithering noise such that the distribution is preserved
    under the :func:`quantize` method.

    The following two models result in identical distributions over ``z``:

        # Model 1.
        z ~ int_dist

        # Model 2.
        x ~ int_dist
        y = dequantize(x, int_dist)
        probs, counts = quantize(y)
        b ~ Bernoulli(probs)
        z = b + counts[..., 0]
    """
    neighbor = count.new_full(count.shape, 0.5)
    neighbor.bernoulli_().mul_(2).add_(-1).add_(count).abs_()

    counts = torch.stack([count, neighbor], dim=-1)
    logits = int_dist.log_prob(count)
    logits[counts == 0] -= math.log(2)
    bern = (logits[..., 1] - logits[..., 0]).exp_().add_(1).reciprocal_().bernoulli_()

    return torch.where(bern, neighbor, count)


class ContinuousBinomial(TorchDistribution):
    """
    Dequantized :class:`~pyro.distributions.Binomial` distribution whose
    ``total_count`` is a positive real number, and whose random value is a
    nonnegative integer.

    This distribution is equivalent to a ``MixtureSameFamily`` over two mixture
    components. If ``n = total_count`` then the two components are:

    - ``Binomial(floor(n), probs)`` with weight ``1 + floor(n) - n``, and
    - ``Binomial(floor(n) + 1, probs)`` with weight ``n - floor(n)``.

    The following two models result in identical distributions over ``z``,
    conditioned on inputs ``n, p``.

        # Model 1.
        b ~ Bernoulli(1 + floor(n) - n)         # Quantize.
        z ~ Binomial(floor(n) + b, p)

        # Model 2.
        z ~ ContinuousBinomial(n, p)

    :param total_count: A tensor of positive real numbers.
    :type total_count: float or torch.Tensor
    :param probs: A tensor of numbers in the unit interval.
    :type probs: float or torch.Tensor
    """
    arg_constraints = {"total_count": constraints.positive,
                       "probs": constraints.unit_interval}
    support = constraints.nonnegative_integer  # Note lack of upper bound.

    def __init__(self, total_count, probs, validate_args=None):
        self.total_count, self.probs, = broadcast_all(total_count, probs)
        self.total_count = self.total_count.type_as(self.probs)
        batch_shape = self.total_count.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ContinuousBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.probs = self.probs.expand(batch_shape)
        super(ContinuousBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def _mixture(self):
        # Compare to MixtureSameFamily after it is released.
        n = self.total_count.unsqueeze(-1)
        lb = n.detach().floor()
        ub = lb + 1
        weights = n - lb
        quantized = torch.cat([lb, ub], dim=-1)
        return weights, quantized

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        weights, total_count = self._mixture
        bern = weights.expand(shape + (1,)).bernoulli()
        lb, ub = total_count.expand(shape + (2,)).unbind(-1)
        total_count = torch.where(bern, ub, lb)
        return Binomial(total_count, self.probs).sample()

    def log_prob(self, value):
        value = value.unsqueeze(-1)
        weights, total_count = self._mixture
        base_dist = Binomial(total_count, self.probs.unsqueeze(-1), validate_args=False)
        log_probs = base_dist.log_prob(value)
        log_probs.masked_fill_(value > total_count, -math.inf)
        logits = torch.cat([(-weights).log1p(), weights.log()], dim=-1)
        log_probs = (log_probs + logits).logsumexp(dim=-1)
        return log_probs


class ContinuousBetaBinomial(TorchDistribution):
    """
    Dequantized :class:`~pyro.distributions.BetaBinomial` distribution whose
    ``total_count`` is a positive real number, and whose random value is a
    nonnegative integer.

    The following three models result in identical distributions over ``z``,
    conditioned on inputs ``c1, c0, n``.

        # Model 1.
        b ~ Bernoulli(1 + floor(n) - n)         # Quantize.
        z ~ BetaBinomial(c1, c0, floor(n) + b)

        # Model 2.
        z ~ ContinuousBetaBinomial(n, p)

    :param concentration1: 1st concentration parameter (alpha) for the Beta
        distribution.
    :type concentration1: float or torch.Tensor
    :param concentration0: 2nd concentration parameter (beta) for the Beta
        distribution.
    :type concentration0: float or torch.Tensor
    :param total_count: A tensor of positive real numbers.
    :type total_count: float or torch.Tensor
    """
    arg_constraints = {"concentration1": constraints.positive,
                       "concentration0": constraints.positive,
                       "total_count": constraints.positive}
    support = constraints.nonnegative_integer  # Note lack of upper bound.

    def __init__(self, concentration1, concentration0, total_count, validate_args=None):
        self.concentration1, self.concentration0, self.total_count = broadcast_all(
            concentration1, concentration0, total_count)
        self.total_count = self.total_count.type_as(self.concentration1)
        batch_shape = self.total_count.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ContinuousBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration1 = self.concentration1.expand(batch_shape)
        new.concentration0 = self.concentration0.expand(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        super(RealRealBetaBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        probs = Beta(self.concentration1, self.concentration0).sample(sample_shape)
        return ContinuousBinomial(self.total_count, probs).sample()

    def log_prob(self, value):
        value = value.unsqueeze(-1)
        weights, total_count = self._mixture
        base_dist = BetaBinomial(self.concentration1.unsqueeze(-1),
                                 self.concentration0.unsqueeze(-1),
                                 total_count, validate_args=False)
        log_probs = base_dist.log_prob(value)
        log_probs.masked_fill_(value > total_count, -math.inf)
        logits = torch.cat([(-weights).log1p(), weights.log()], dim=-1)
        log_probs = (log_probs + logits).logsumexp(dim=-1)
        return log_probs


class DequantizedDistribution(TorchDistribution):
    """
    Dequantized distribution.

    Given a distribution ``int_dist`` over nonnegative integers, this creates a
    distribution over positive numbers such that randomly rounded samples from
    this distribution will be distributed according to ``int_dist``.

    For example the following three models result in identical distributions
    over ``z``, conditioned on inputs ``n, p``.

        # Model 1.
        z ~ Binomial(n, p)

        # Model 2.
        r ~ DequantizedDistribution(Binomial(n, p))
        b ~ Bernoulli(1 + floor(r) - r)      # Quantize.
        z = floor(r) + b

    """
    arg_constraints = {}
    support = constraints.positive  # Note lack of upper bound.

    def __init__(self, int_dist, validate_args=None):
        self.int_dist = int_dist
        super().__init__(int_dist.batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ContinuousBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.int_dist = self.int_dist.expand(batch_shape)
        super(DequantizedDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        int_value = self.int_dist.sample()
        return dequantize(int_value, self.int_dist)

    # TODO implement a separate .rsample().

    def log_prob(self, value):
        # FIXME
        value = value.unsqueeze(-1)
        weights, counts = quantize(value)

        # The first part is a mixture model.
        lb = self.total_count.floor().long()
        total_count = torch.stack([lb, ub + 1], dim=-1)
        logits = torch.stack([ub - self.total_count, self.total_count - lb], dim=-1).log()
        log_prob = dist.BetaBinomial(self.concentration0.unsqeeze(-1),
                                     self.concentration1.unsqeeze(-1),
                                     total_count).log_prob(value.unsqueeze(-1))
        # Unbounded support is required by actual applications.
        log_prob.masked_fill_(value.unsqueeze(-1) > ub, float(-inf))
        return (logits + log_prob).logsumexp(dim=-1)  # mixture

    @property
    def mean(self):
        return self.int_dist.mean
