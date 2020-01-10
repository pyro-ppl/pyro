# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from pyro.distributions.torch_distribution import TorchDistribution


def _check(x):
    assert not torch.isnan(x).any()
    return x


def _unsafe_standard_stable(alpha, beta, V, W):
    # Implements a noisily reparametrized version of the sampler
    # Chambers-Mallows-Stuck method as corrected by Weron [1,3] and simplified
    # by Nolan [4]. This will fail if alpha is close to 1.

    # Differentiably transform noise via parameters.
    assert V.shape == W.shape
    inv_alpha = alpha.reciprocal()
    b = beta * (math.pi / 2 * alpha).tan()
    v = b.atan() + alpha * V
    Z = v.sin() / ((1 + b * b).rsqrt() * V.cos()).pow(inv_alpha) \
        * ((v - V).cos() / W).pow(inv_alpha - 1)
    Z.data[~(Z.data == Z.data)] = 0  # drop occasional NANs

    # Convert to Nolan's parametrization S^0 so that samples depend
    # continuously on (alpha,beta), allowing us to interpolate around the hole
    # at alpha=1.
    X = Z - b
    return X, Z


RADIUS = 0.01


def _standard_stable(alpha, beta, aux_uniform, aux_exponential):
    """
    Differentiably transform two random variables::

        aux_uniform ~ Uniform(-pi/2, pi/2)
        aux_exponential ~ Exponential(1)

    to a standard ``Stable(alpha, beta)`` random variable.
    """
    # Determine whether a hole workaround is needed.
    with torch.no_grad():
        hole = 1.
        near_hole = (alpha - hole).abs() <= RADIUS
    if not torch._C._get_tracing_state() and not near_hole.any():
        x, _ = _unsafe_standard_stable(alpha, beta, aux_uniform, aux_exponential)
        return x

    # Avoid the hole at alpha=1 by interpolating between pairs
    # of points at hole-RADIUS and hole+RADIUS.
    aux_uniform_ = aux_uniform.unsqueeze(-1)
    aux_exponential_ = aux_exponential.unsqueeze(-1)
    beta_ = beta.unsqueeze(-1)
    alpha_ = alpha.unsqueeze(-1).expand(alpha.shape + (2,)).contiguous()
    with torch.no_grad():
        lower, upper = alpha_.unbind(-1)
        lower.data[near_hole] = hole - RADIUS
        upper.data[near_hole] = hole + RADIUS
        # We don't need to backprop through weights, since we've pretended
        # alpha_ is reparametrized, even though we've clamped some values.
        #               |a - a'|
        # weight = 1 - ----------
        #              2 * RADIUS
        weights = (alpha_ - alpha.unsqueeze(-1)).abs_().mul_(-1 / (2 * RADIUS)).add_(1)
        weights[~near_hole] = 0.5
    pairs, _ = _unsafe_standard_stable(alpha_, beta_, aux_uniform_, aux_exponential_)
    return (pairs * weights).sum(-1)


class Stable(TorchDistribution):
    r"""
    Levy :math:`\alpha`-stable distribution. See [1] for a review.

    This uses Nolan's parametrization [2] of the ``loc`` parameter, which is
    required for continuity and differentiability. This corresponds to the
    notation :math:`S^0_\alpha(\beta,\sigma,\mu_0)` of [1], where
    :math:`\alpha` = stability, :math:`\beta` = skew, :math:`\sigma` = scale,
    and :math:`\mu_0` = loc.

    This implements a reparametrized sampler :meth:`rsample` , but does not
    implement :meth:`log_prob` . Inference can be performed using either
    likelihood-free algorithms such as
    :class:`~pyro.infer.energy_distance.EnergyDistance`, or reparameterization
    via the :func:`~pyro.poutine.handlers.reparam` handler with
    :class:`~pyro.infer.reparam.stable.StableReparam` e.g.::

        with poutine.reparam():
            pyro.sample("x", Stable(stability, skew, scale, loc),
                        infer={"reparam": StableReparam()})

    [1] S. Borak, W. Hardle, R. Weron (2005).
        Stable distributions.
        https://edoc.hu-berlin.de/bitstream/handle/18452/4526/8.pdf
    [2] J.P. Nolan (1997).
        Numerical calculation of stable densities and distribution functions.
    [3] Rafal Weron (1996).
        On the Chambers-Mallows-Stuck Method for
        Simulating Skewed Stable Random Variables.
    [4] J.P. Nolan (2017).
        Stable Distributions: Models for Heavy Tailed Data.
        http://fs2.american.edu/jpnolan/www/stable/chap1.pdf

    :param Tensor stability: Levy stability parameter :math:`\alpha\in(0,2]` .
    :param Tensor skew: Skewness :math:`\beta\in[-1,1]` .
    :param Tensor scale: Scale :math:`\sigma > 0` . Defaults to 1.
    :param Tensor loc: Location :math:`\mu_0` in Nolan's parametrization [2].
        Defaults to 0.
    """
    has_rsample = True
    arg_constraints = {"stability": constraints.interval(0, 2),  # half-open (0, 2]
                       "skew": constraints.interval(-1, 1),  # closed [-1, 1]
                       "scale": constraints.positive,
                       "loc": constraints.real}
    support = constraints.real

    def __init__(self, stability, skew, scale=1.0, loc=0.0, validate_args=None):
        self.stability, self.skew, self.scale, self.loc = broadcast_all(
            stability, skew, scale, loc)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Stable, _instance)
        batch_shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        super(Stable, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        raise NotImplementedError("Stable.log_prob() is not implemented")

    def rsample(self, sample_shape=torch.Size()):
        # Draw parameter-free noise.
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            new_empty = self.stability.new_empty
            aux_uniform = new_empty(shape).uniform_(-math.pi / 2, math.pi / 2)
            aux_exponential = new_empty(shape).exponential_()

        # Differentiably transform.
        x = _standard_stable(self.stability, self.skew, aux_uniform, aux_exponential)
        return self.loc + self.scale * x
