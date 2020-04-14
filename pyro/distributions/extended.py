# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions.utils import broadcast_all, lazy_property
from torch.distributions import constraints

from .conjugate import BetaBinomial
from .torch import Beta, Binomial
from .torch_distribution import TorchDistribution


class ExtendedBinomial(TorchDistribution):
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
        z ~ ExtendedBinomial(n, p)

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
        new = self._get_checked_instance(ExtendedBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.probs = self.probs.expand(batch_shape)
        super(ExtendedBinomial, new).__init__(batch_shape, validate_args=False)
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


class ExtendedBetaBinomial(TorchDistribution):
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
        z ~ ExtendedBetaBinomial(n, p)

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
        new = self._get_checked_instance(ExtendedBetaBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration1 = self.concentration1.expand(batch_shape)
        new.concentration0 = self.concentration0.expand(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        super(ExtendedBetaBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        probs = Beta(self.concentration1, self.concentration0).sample(sample_shape)
        return ExtendedBinomial(self.total_count, probs).sample()

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
