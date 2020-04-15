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
    Extension of a :class:`~pyro.distributions.Binomial` distribution to allow
    positive real ``total_count`` and unbounded nonnegative integer samples.

    This agrees in distribution with a ``MixtureSameFamily`` over two mixture
    components. If ``n = total_count`` then the two components are:

    - ``Binomial(floor(n), probs)`` with weight ``1 + floor(n) - n``, and
    - ``Binomial(floor(n) + 1, probs)`` with weight ``n - floor(n)``.

    The following two models result in identical distributions over ``z``,
    conditioned on inputs ``n, p``.

        # Model 1.
        b ~ Bernoulli(n - floor(n))    # Quantize.
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
        n = self.total_count.unsqueeze(-1)
        lb = n.detach().floor()
        ub = lb + 1
        weights = torch.cat([ub - n, n - lb], dim=-1)
        quantized = torch.cat([lb, ub], dim=-1)
        return weights, quantized

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        weights, quantized = self._mixture
        bern = weights[..., 1].expand(shape).bernoulli()
        total_count = quantized[..., 0] + bern
        return Binomial(total_count, self.probs).sample()

    def log_prob(self, value):
        value = value.unsqueeze(-1)
        weights, quantized = self._mixture
        log_prob = Binomial(quantized, self.probs.unsqueeze(-1),
                            validate_args=False).log_prob(value)
        log_prob = log_prob.masked_fill(value > quantized, -math.inf)
        log_prob = (log_prob + weights.log()).logsumexp(dim=-1)
        return log_prob


class ExtendedBetaBinomial(TorchDistribution):
    """
    Extension of a :class:`~pyro.distributions.BetaBinomial` distribution to
    allow positive real ``total_count`` and unbounded nonnegative integer
    samples.

    The following three models result in identical distributions over ``z``,
    conditioned on inputs ``c1, c0, n``.

        # Model 1.
        b ~ Bernoulli(n + floor(n))             # Quantize.
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

    @lazy_property
    def _mixture(self):
        n = self.total_count.unsqueeze(-1)
        lb = n.detach().floor()
        ub = lb + 1
        weights = torch.cat([ub - n, n - lb], dim=-1)
        quantized = torch.cat([lb, ub], dim=-1)
        return weights, quantized

    def sample(self, sample_shape=torch.Size()):
        probs = Beta(self.concentration1, self.concentration0).sample(sample_shape)
        return ExtendedBinomial(self.total_count, probs).sample()

    def log_prob(self, value):
        value = value.unsqueeze(-1)
        weights, quantized = self._mixture
        log_prob = BetaBinomial(self.concentration1.unsqueeze(-1),
                                self.concentration0.unsqueeze(-1),
                                quantized).log_prob(value)
        log_prob = log_prob.masked_fill(value > quantized, -math.inf)
        log_prob = (log_prob + weights.log()).logsumexp(dim=-1)
        return log_prob
