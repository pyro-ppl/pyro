# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from pyro.ops.special import log_beta

from .torch_distribution import TorchDistribution

# TODO remove this after PyTorch 1.6 is released.
if hasattr(torch, "logaddexp"):
    _logaddexp = torch.logaddexp
else:
    def _logaddexp(x, y):
        if torch._C._get_tracing_state() or x.shape != y.shape:
            x, y = torch.broadcast_tensors(x, y)
        return torch.stack([x, y], dim=-1).logsumexp(dim=-1)


class RelaxedBetaBinomial(TorchDistribution):
    """
    EXPERIMENTAL :class:`~pyro.distributions.BetaBinomial` distribution relaxed
    such that the support is the entire real line and the ``total_count``
    parameter can have arbitrary real value.

    This is useful for approximating discrete models as continuous models for
    inference via :class:`~pyro.infer.mcmc.HMC` or
    :class:`~pyro.infer.svi.SVI`.

    This approximates a :class:`~pyro.distributions.BetaBinomial` distribution
    as a mixture of a :class:`~pyro.distributions.Beta` (with bounded support)
    and a :class:`~pyro.distributions.Normal` (with infinite support), both of
    which are moment-matched to the original distribution.  To ensure values
    and gradients are well-defined, the Beta's concentration is lower bounded
    by 2 if gradients are required (if ``value.requires_grad``) or by 1 if
    gradients are not required.

    :param total_count: Number of Bernoulli trials.
    :type total_count: float or torch.Tensor
    :param mean: Mean of the distribution.
    :type mean: float or torch.Tensor
    :param variance: Variance of the unrelaxed distribution. Actual mean will be
    :type variance: float or torch.Tensor
    """
    arg_constraints = {"total_count": constraints.real,
                       "mean": constraints.real,
                       "variance": constraints.real}
    support = constraints.real

    def __init__(self, total_count, mean, variance, *,
                 normal_weight=0.1, validate_args=None):
        total_count, mean, variance = broadcast_all(total_count, mean, variance)
        # Clamp to ensure feasibility.
        self.total_count = total_count.clamp(min=0)
        self._mean = torch.min(total_count, mean.clamp(min=0))
        variance = variance.clamp(min=0)
        # Inflate variance by Uniform(0,1).variance, corresponding to the
        # relaxation from integer grid points to continuous values.
        self._variance = variance + 1/12
        batch_shape = self.total_count.shape,
        event_shape = ()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        assert 0 < normal_weight < 1
        self.normal_weight = normal_weight

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        new = self._get_checked_instance(RelaxedBetaBinomial, _instance)
        new.total_count = self.total_count.expand(batch_shape)
        new._mean = self.mean
        new._variance = self.variance
        new.normal_weight = self.normal_weight
        super(RelaxedBetaBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        eps = torch.finfo(value.dtype).eps
        total_count, mean, variance = self.total_count, self.mean, self.variance

        # Construct a moment-matched Normal distribution.
        normal_part = (-0.5) * ((value - mean) ** 2 / variance + variance.log()
                                + math.log(2 * math.pi))

        # Construct a moment-matched Beta distribution over a widened interval
        # [-1/2, total_count + 1/2].
        n = total_count + 1
        x = (value + 0.5) / n
        p = (mean + 0.5) / n
        q = 1 - p
        concentration = p * q * n ** 2 / variance - 1
        # Lower-bound the concentration to ensure finite value and optionally
        # finite gradient wrt value. This form of clamping preserves the mean.
        min_concentration = 2 if value.requires_grad else 1
        concentration = torch.max(concentration, min_concentration / torch.min(p, q))
        c1 = p * concentration
        c0 = q * concentration
        mask = (0 < x) & (x < 1)
        x = x.clamp(min=eps, max=1 - eps)
        beta_part = ((c1 - 1) * x.log() + (c0 - 1) * (-x).log1p()
                     - log_beta(c1, c0, tol=0.01) - n.log())
        beta_part = torch.where(mask, beta_part, torch.tensor(-math.inf))

        # Mix the Normal and Beta together.
        w0 = self.normal_weight
        w1 = 1 - w0
        return _logaddexp(normal_part + math.log(w0), beta_part + math.log(w1))
