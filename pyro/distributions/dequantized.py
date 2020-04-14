# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints

from .torch_distribution import TorchDistribution


class DequantizedDistribution(TorchDistribution):
    """
    Dequantized distribution over relaxed counts.

    The density of this distribution linearly interpolates the density of
    ``base_dist``, except in the interval ``[0,1]`` where the weight of 0 is
    doubled to account for asymmetry.

    Given a distribution ``base_dist`` over nonnegative integers, this creates
    a distribution over positive reals such that randomly rounded samples from
    this distribution will be distributed approximately according to
    ``base_dist``.

    For example the following two models result in approximately identical
    distributions over ``z``, conditioned on inputs ``n, p``.

        # Model 1.
        z ~ Binomial(n, p)

        # Model 2.
        r ~ DequantizedDistribution(Binomial(n, p))
        b ~ Bernoulli(1 + floor(r) - r)      # Quantize.
        z = floor(r) + b

    Earth mover distance error is upper bounded by 1/2.

    :param base_dist: A distribution with
        ``.support == constraints.nonnegative_integer`` and
        ``.event_shape == ()``.
    :type base_dist: ~pyro.distributions.TorchDistribution
    """
    arg_constraints = {}
    support = constraints.positive  # Note lack of upper bound.
    has_rsample = False  # TODO implement .rsample() based on base_dist.cdf().

    def __init__(self, base_dist, validate_args=None):
        self.base_dist = base_dist
        super().__init__(base_dist.batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DequantizedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.base_dist = self.base_dist.expand(batch_shape)
        super(DequantizedDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        value = self.base_dist.sample(sample_shape)
        # Add uniform triangular noise in [-1,1]
        noise = value.new_empty(value.shape).uniform_(-0.25, 0.25)
        sign = noise.sign()
        noise.abs_().sqrt_().add_(-0.5).mul_(sign).add_(0.5)
        # Ensure result is nonnegative.
        return noise.add_(value).abs_()

    def log_prob(self, value):
        missing_dims = len(self.batch_shape) - value.dim()
        if missing_dims:
            value = value.reshape((1,) * missing_dims + value.shape)
        with torch.no_grad():
            lb = value.floor()
            ub = lb + 1
            quantized = torch.stack([lb, ub])

        log_prob = self.base_dist.log_prob(quantized)
        # Account for asymmetry at the interval [0,1].
        log_prob = torch.where(quantized == 0, log_prob, log_prob + math.log(2))
        # Allow unbounded support (required by many applications).
        log_prob = log_prob.masked_fill(quantized > ub, -math.inf)
        logits = torch.stack([ub - value, value - lb]).log()
        return (logits + log_prob).logsumexp(dim=0)  # Mixture.
