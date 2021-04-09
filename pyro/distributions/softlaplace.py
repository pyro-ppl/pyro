# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from .torch_distribution import TorchDistribution


class SoftLaplace(TorchDistribution):
    """
    Smooth distribution with Laplace-like tail behavior.

    This distribution corresponds to the log-convex density::

        z = (value - loc) / scale
        log_prob = log(2 / pi) - log(scale) - logaddexp(z, -z)

    Like the Laplace density, this density has the heaviest possible tails
    (asymptotically) while still being log-convex. Unlike the Laplace
    distribution, this distribution is infinitely differentiable everywhere,
    and is thus suitable for constructing Laplace approximations.

    :param loc: Location parameter.
    :param scale: Scale parameter.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, *, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SoftLaplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(SoftLaplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return math.log(2 / math.pi) - self.scale.log() - torch.logaddexp(z, -z)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self.loc.new_empty(shape).uniform_()
        return self.icdf(u)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return z.exp().atan().mul(2 / math.pi)

    def icdf(self, value):
        return value.mul(math.pi / 2).tan().log().mul(self.scale).add(self.loc)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (math.pi / 2 * self.scale) ** 2
