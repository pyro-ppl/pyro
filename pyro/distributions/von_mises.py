from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from pyro.distributions import TorchDistribution


def _eval_poly(y, coef):
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
_COEF_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
               -0.2057706e-1, 0.2635537e-1, -0.1647633e-1,  0.392377e-2]


def _log_modified_bessel_fn_0(x):
    """
    Returns ``log(I0(x))`` for ``x > 0``.
    """
    # compute small solution
    y = (x / 3.75).pow(2)
    small = _eval_poly(y, _COEF_SMALL).log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE).log()

    mask = (x < 3.75)
    result = large
    if mask.any():
        result[mask] = small[mask]
    return result


class VonMises(TorchDistribution):
    """
    A circular von Mises distribution.

    Currently only :meth:`log_prob` is implemented.

    :param torch.Tensor loc: an angle in radians.
    :param torch.Tensor concentration: concentration parameter
    """
    arg_constraints = {'loc': constraints.real, 'concentration': constraints.positive}
    support = constraints.real

    def __init__(self, loc, concentration, validate_args=None):
        self.loc, self.concentration = broadcast_all(loc, concentration)
        batch_shape = self.loc.shape
        event_shape = torch.Size()
        super(VonMises, self).__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        log_prob = self.concentration * torch.cos(value - self.loc)
        log_prob = log_prob - math.log(2 * math.pi) - _log_modified_bessel_fn_0(self.concentration)
        return log_prob

    def expand(self, batch_shape):
        validate_args = self.__dict__.get('validate_args')
        loc = self.loc.expand(batch_shape)
        concentration = self.concentration.expand(batch_shape)
        return VonMises(loc, concentration, validate_args=validate_args)
