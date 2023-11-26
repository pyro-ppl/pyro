# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.nn.functional import logsigmoid

from .torch_distribution import TorchDistribution


class Logistic(TorchDistribution):
    r"""
    Logistic distribution.

    This is a smooth distribution with symmetric asymptotically exponential
    tails and a concave log density. For standard ``loc=0``, ``scale=1``, the
    density is given by

    .. math::

        p(x) = \frac {e^{-x}} {(1 + e^{-x})^2}

    Like the :class:`~pyro.distributions.Laplace` density, this density has the
    heaviest possible tails (asymptotically) while still being log-convex.
    Unlike the :class:`~pyro.distributions.Laplace` distribution, this
    distribution is infinitely differentiable everywhere, and is thus suitable
    for constructing Laplace approximations.

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
        new = self._get_checked_instance(Logistic, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Logistic, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return logsigmoid(z) * 2 - z - self.scale.log()

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self.loc.new_empty(shape).uniform_()
        return self.icdf(u)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return z.sigmoid()

    def icdf(self, value):
        return self.loc + self.scale * value.logit()

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale**2 * (math.pi**2 / 3)

    def entropy(self):
        return self.scale.log() + 2


class SkewLogistic(TorchDistribution):
    r"""
    Skewed generalization of the Logistic distribution (Type I in [1]).

    This is a smooth distribution with asymptotically exponential tails and a
    concave log density. For standard ``loc=0``, ``scale=1``, ``asymmetry=α``
    the density is given by

    .. math::

        p(x;\alpha) = \frac {\alpha e^{-x}} {(1 + e^{-x})^{\alpha+1}}

    Like the :class:`~pyro.distributions.AsymmetricLaplace` density, this
    density has the heaviest possible tails (asymptotically) while still being
    log-convex. Unlike the :class:`~pyro.distributions.AsymmetricLaplace`
    distribution, this distribution is infinitely differentiable everywhere,
    and is thus suitable for constructing Laplace approximations.

    **References**

    [1] Generalized logistic distribution
        https://en.wikipedia.org/wiki/Generalized_logistic_distribution

    :param loc: Location parameter.
    :param scale: Scale parameter.
    :param asymmetry: Asymmetry parameter (positive). The distribution skews
        right when ``asymmetry > 1`` and left when ``asymmetry < 1``. Defaults
        to ``asymmetry = 1`` corresponding to the standard Logistic
        distribution.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "asymmetry": constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, asymmetry=1.0, *, validate_args=None):
        self.loc, self.scale, self.asymmetry = broadcast_all(loc, scale, asymmetry)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SkewLogistic, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.asymmetry = self.asymmetry.expand(batch_shape)
        super(SkewLogistic, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        a = self.asymmetry
        return a.log() - z + logsigmoid(z) * (a + 1) - self.scale.log()

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self.loc.new_empty(shape).uniform_()
        return self.icdf(u)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return z.sigmoid().pow(self.asymmetry)

    def icdf(self, value):
        z = value.pow(self.asymmetry.reciprocal()).logit()
        return self.loc + self.scale * z
