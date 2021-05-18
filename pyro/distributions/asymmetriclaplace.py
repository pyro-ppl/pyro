# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from .torch_distribution import TorchDistribution


class AsymmetricLaplace(TorchDistribution):
    """
    Asymmetric version of Laplace distribution.

    To the left of ``loc`` this acts like an ``-Exponential(left_scale)``;
    to the right of ``loc`` this acts like an ``Exponential(right_scale)``.
    The density is continuous so the left and right densities at ``loc`` agree.

    :param loc: Location parameter, i.e. the mode.
    :param left_scale: Scale parameter to the left of ``loc``.
    :param right_scale: Scale parameter to the right of ``loc``.
    """
    arg_constraints = {"loc": constraints.real,
                       "right_scale": constraints.positive,
                       "left_scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, left_scale, right_scale, validate_args=None):
        self.loc, self.left_scale, self.right_scale = broadcast_all(
            loc, left_scale, right_scale
        )
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(AsymmetricLaplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.left_scale = self.left_scale.expand(batch_shape)
        new.right_scale = self.right_scale.expand(batch_shape)
        super(AsymmetricLaplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = value - self.loc
        z = -z.abs() / torch.where(z < 0, self.left_scale, self.right_scale)
        return z - (self.left_scale + self.right_scale).log()

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        denom = self.left_scale + self.right_scale
        shift = -self.left_scale / denom
        u = self.loc.new_empty(shape).uniform_() - shift
        scale = torch.where(u < 0, -self.left_scale, self.right_scale)
        u = u.abs().mul(denom / scale)
        u = u.data.clamp(max=1 - torch.finfo(u.dtype).eps)
        return u.neg().log1p().mul(scale)
