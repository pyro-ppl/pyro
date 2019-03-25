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


_I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
_I0_COEF_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                  -0.2057706e-1, 0.2635537e-1, -0.1647633e-1,  0.392377e-2]
_I1_COEF_SMALL = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.2658733e-1, 0.301532e-2, 0.32411e-3]
_I1_COEF_LARGE = [0.39894228, -0.3988024e-1, -0.362018e-2, 0.163801e-2, -0.1031555e-1,
                  0.2282967e-1, -0.2895312e-1, 0.1787654e-1, -0.420059e-2]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0):
    """
    Returns ``log(I_order(x))`` for ``x > 0``,
    where `order` is either 0 or 1.
    """
    assert order == 0 or order == 1

    # compute small solution
    y = (x / 3.75).pow(2)
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    mask = (x < 3.75)
    result = large
    if mask.any():
        result[mask] = small[mask]
    return result


class VonMises(TorchDistribution):
    """
    A circular von Mises distribution.

    This implementation uses polar coordinates. The ``loc`` and ``value`` args
    can be any real number (to facilitate unconstrained optimization), but are
    interpreted as angles modulo 2 pi.

    See :class:`~pyro.distributions.VonMises3D` for a 3D cartesian coordinate
    cousin of this distribution.

    :param torch.Tensor loc: an angle in radians.
    :param torch.Tensor concentration: concentration parameter
    """
    arg_constraints = {'loc': constraints.real, 'concentration': constraints.positive}
    support = constraints.real
    has_rsample = False

    def __init__(self, loc, concentration, validate_args=None):
        self.loc, self.concentration = broadcast_all(loc, concentration)
        batch_shape = self.loc.shape
        event_shape = torch.Size()

        # Moments
        self._variance = 1 - (_log_modified_bessel_fn(self.concentration, order=1) -
                              _log_modified_bessel_fn(self.concentration, order=0)).exp()

        # Parameters for sampling
        tau = 1 + (1 + 4 * self.concentration ** 2).sqrt()
        rho = (tau - (2 * tau).sqrt()) / (2 * self.concentration)
        self._proposal_r = (1 + rho ** 2) / (2 * rho)

        super(VonMises, self).__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        log_prob = self.concentration * torch.cos(value - self.loc)
        log_prob = log_prob - math.log(2 * math.pi) - _log_modified_bessel_fn(self.concentration, order=0)
        return log_prob

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        """
        The sampling algorithm for the von Mises distribution is based on the following paper:
        Best, D. J., and Nicholas I. Fisher.
        "Efficient simulation of the von Mises distribution." Applied Statistics (1979): 152-157.
        """
        shape = self._extended_shape(sample_shape)
        x = torch.empty(shape, dtype=self.loc.dtype, device=self.loc.device)
        done = torch.zeros(shape, dtype=self.loc.dtype, device=self.loc.device).byte()
        while not done.all():
            u = torch.rand((3,) + shape, dtype=self.loc.dtype, device=self.loc.device)
            u1, u2, u3 = u.unbind()
            z = torch.cos(math.pi * u1)
            f = (1 + self._proposal_r * z) / (self._proposal_r + z)
            c = self.concentration * (self._proposal_r - f)
            accept = ((c * (2 - c) - u2) > 0) | ((c / u2).log() + 1 - c >= 0)
            if accept.any():
                x[accept] = torch.sign(u3[accept] - 0.5) * torch.acos(f[accept])
                done |= accept
        return (x + math.pi + self.loc) % (2 * math.pi) - math.pi

    def expand(self, batch_shape):
        try:
            return super(VonMises, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            loc = self.loc.expand(batch_shape)
            concentration = self.concentration.expand(batch_shape)
            return type(self)(loc, concentration, validate_args=validate_args)

    @property
    def mean(self):
        """
        The provided mean is the circular one.
        """
        return self.loc

    @property
    def variance(self):
        """
        The provided variance is the circular one.
        """
        return self._variance
