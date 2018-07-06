from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import constraints

from pyro.distributions import TorchDistribution


class VonMises3D(TorchDistribution):
    """
    Spherical von Mises distribution.
    https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    """
    arg_constraints = {'concentration': constraints.real}
    support = constraints.real  # TODO implement constraints.sphere or similar

    def __init__(self, concentration, validate_args=None):
        if concentration.dim() < 1 or concentration.shape[-1] != 3:
            raise ValueError('Expected concentration to have rightmost dim 3, actual shape = {}'.format(
                concentration.shape))
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super(VonMises3D, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            if value.dim() < 1 or value.shape[-1] != 3:
                raise ValueError('Expected value to have rightmost dim 3, actual shape = {}'.format(
                    value.shape))
            if not (torch.abs(value.norm(2, -1) - 1) < 1e-6).all():
                raise ValueError('direction vectors are not normalized')
        scale = self.concentration.norm(2, -1)
        log_normalizer = scale.log() - scale.sinh().log() - math.log(4 * math.pi)
        return (self.concentration * value).sum(-1) + log_normalizer

    def expand(self, batch_shape):
        try:
            return super(VonMises3D, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            concentration = self.concentration.expand(torch.Size(batch_shape) + (3,))
            return type(self)(concentration, validate_args=validate_args)
