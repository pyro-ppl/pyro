# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property

from .torch_distribution import TorchDistribution


class AsymmetricLaplace(TorchDistribution):
    """
    Asymmetric version of Laplace distribution.

    To the left of ``loc`` this acts like an
    ``-Exponential(1/(asymmetry*scale))``; to the right of ``loc`` this acts
    like an ``Exponential(asymmetry/scale)``. The density is continuous so the
    left and right densities at ``loc`` agree.

    :param loc: Location parameter, i.e. the mode.
    :param scale: Scale parameter = geometric mean of left and right scales.
    :param asymmetry: Square of ratio of left to right scales.
    """
    arg_constraints = {"loc": constraints.real,
                       "scale": constraints.positive,
                       "asymmetry": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, asymmetry, validate_args=None):
        self.loc, self.scale, self.asymmetry = broadcast_all(loc, scale, asymmetry)
        super().__init__(self.loc.shape, validate_args=validate_args)

    @lazy_property
    def left_scale(self):
        return self.scale * self.asymmetry

    @lazy_property
    def right_scale(self):
        return self.scale / self.asymmetry

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(AsymmetricLaplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.asymmetry = self.asymmetry.expand(batch_shape)
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
        u, v = self.loc.new_empty((2,) + shape).exponential_()
        return self.loc - self.left_scale * u + self.right_scale * v

    @property
    def mean(self):
        total_scale = self.left_scale + self.right_scale
        return self.loc + (self.right_scale ** 2 - self.left_scale ** 2) / total_scale

    @property
    def variance(self):
        left = self.left_scale
        right = self.right_scale
        total = left + right
        p = left / total
        q = right / total
        return p * left ** 2 + q * right ** 2 + p * q * total ** 2
