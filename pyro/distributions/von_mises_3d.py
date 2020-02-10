# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints

from pyro.distributions import TorchDistribution


class VonMises3D(TorchDistribution):
    """
    Spherical von Mises distribution.

    This implementation combines the direction parameter and concentration
    parameter into a single combined parameter that contains both direction and
    magnitude. The ``value`` arg is represented in cartesian coordinates: it
    must be a normalized 3-vector that lies on the 2-sphere.

    See :class:`~pyro.distributions.VonMises` for a 2D polar coordinate cousin
    of this distribution.

    Currently only :meth:`log_prob` is implemented.

    :param torch.Tensor concentration: A combined location-and-concentration
        vector. The direction of this vector is the location, and its
        magnitude is the concentration.
    """
    arg_constraints = {'concentration': constraints.real}
    support = constraints.real  # TODO implement constraints.sphere or similar

    def __init__(self, concentration, validate_args=None):
        if concentration.dim() < 1 or concentration.shape[-1] != 3:
            raise ValueError('Expected concentration to have rightmost dim 3, actual shape = {}'.format(
                concentration.shape))
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

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
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            concentration = self.concentration.expand(torch.Size(batch_shape) + (3,))
            return type(self)(concentration, validate_args=validate_args)
