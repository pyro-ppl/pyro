# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.transforms import TanhTransform, Transform

# TODO: Move upstream


class ELUTransform(Transform):
    r"""
    Bijective transform via the mapping :math:`y = \text{ELU}(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, ELUTransform)

    def _call(self, x):
        return F.elu(x)

    def _inverse(self, y, eps=1e-8):
        return torch.max(y, torch.zeros_like(y)) + torch.min(torch.log1p(y + eps), torch.zeros_like(y))

    def log_abs_det_jacobian(self, x, y):
        return -F.relu(-x)


def elu():
    """
    A helper function to create an
    :class:`~pyro.distributions.transform.ELUTransform` object for consistency with
    other helpers.
    """
    return ELUTransform()

# TODO: Move upstream


class LeakyReLUTransform(Transform):
    r"""
    Bijective transform via the mapping :math:`y = \text{LeakyReLU}(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, LeakyReLUTransform)

    def _call(self, x):
        return F.leaky_relu(x)

    def _inverse(self, y):
        return F.leaky_relu(y, negative_slope=100.0)

    def log_abs_det_jacobian(self, x, y):
        return torch.where(x >= 0., torch.zeros_like(x), torch.ones_like(x) * math.log(0.01))


def leaky_relu():
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.LeakyReLUTransform` object for
    consistency with other helpers.
    """
    return LeakyReLUTransform()


def tanh():
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.TanhTransform` object for consistency
    with other helpers.
    """
    return TanhTransform()
