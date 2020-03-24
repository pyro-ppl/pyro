# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.stable import _standard_stable, _unsafe_standard_stable
from pyro.infer.util import is_validation_enabled

from .reparam import Reparam


class LatentStableReparam(Reparam):
    """
    Auxiliary variable reparameterizer for
    :class:`~pyro.distributions.Stable` latent variables.

    This is useful in inference of latent :class:`~pyro.distributions.Stable`
    variables because the :meth:`~pyro.distributions.Stable.log_prob` is not
    implemented.

    This uses the Chambers-Mallows-Stuck method [1], creating a pair of
    parameter-free auxiliary distributions (``Uniform(-pi/2,pi/2)`` and
    ``Exponential(1)``) with well-defined ``.log_prob()`` methods, thereby
    permitting use of reparameterized stable distributions in likelihood-based
    inference algorithms like SVI and MCMC.

    This reparameterization works only for latent variables, not likelihoods.
    For likelihood-compatible reparameterization see
    :class:`SymmetricStableReparam` or :class:`StableReparam` .

    [1] J.P. Nolan (2017).
        Stable Distributions: Models for Heavy Tailed Data.
        http://fs2.american.edu/jpnolan/www/stable/chap1.pdf
    """
    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.Stable) and fn.coords == "S0"
        assert obs is None, "LatentStableReparam does not support observe statements"

        # Draw parameter-free noise.
        proto = fn.stability
        half_pi = proto.new_tensor(math.pi / 2)
        one = proto.new_ones(proto.shape)
        u = pyro.sample("{}_uniform".format(name),
                        self._wrap(dist.Uniform(-half_pi, half_pi).expand(proto.shape), event_dim))
        e = pyro.sample("{}_exponential".format(name),
                        self._wrap(dist.Exponential(one), event_dim))

        # Differentiably transform.
        x = _standard_stable(fn.stability, fn.skew, u, e, coords="S0")
        value = fn.loc + fn.scale * x

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=event_dim).mask(False)
        return new_fn, value


class SymmetricStableReparam(Reparam):
    """
    Auxiliary variable reparameterizer for symmetric
    :class:`~pyro.distributions.Stable` random variables (i.e. those for which
    ``skew=0``).

    This is useful in inference of symmetric
    :class:`~pyro.distributions.Stable` variables because the
    :meth:`~pyro.distributions.Stable.log_prob` is not implemented.

    This reparameterizes a symmetric :class:`~pyro.distributions.Stable` random
    variable as a totally-skewed (``skew=1``)
    :class:`~pyro.distributions.Stable` scale mixture of
    :class:`~pyro.distributions.Normal` random variables. See Proposition 3. of
    [1] (but note we differ since :class:`Stable` uses Nolan's continuous S0
    parameterization).

    [1] Alvaro Cartea and Sam Howison (2009)
        "Option Pricing with Levy-Stable Processes"
        https://pdfs.semanticscholar.org/4d66/c91b136b2a38117dd16c2693679f5341c616.pdf
    """
    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.Stable) and fn.coords == "S0"
        if is_validation_enabled():
            if not (fn.skew == 0).all():
                raise ValueError("SymmetricStableReparam found nonzero skew")
            if not (fn.stability < 2).all():
                raise ValueError("SymmetricStableReparam found stability >= 2")

        # Draw parameter-free noise.
        proto = fn.stability
        half_pi = proto.new_tensor(math.pi / 2)
        one = proto.new_ones(proto.shape)
        u = pyro.sample("{}_uniform".format(name),
                        self._wrap(dist.Uniform(-half_pi, half_pi).expand(proto.shape), event_dim))
        e = pyro.sample("{}_exponential".format(name),
                        self._wrap(dist.Exponential(one), event_dim))

        # Differentiably transform to scale drawn from a totally-skewed stable variable.
        a = fn.stability
        z = _unsafe_standard_stable(a / 2, 1, u, e, coords="S")
        assert (z >= 0).all()
        scale = fn.scale * (math.pi / 4 * a).cos().pow(a.reciprocal()) * z.sqrt()
        scale = scale.clamp(min=torch.finfo(scale.dtype).tiny)

        # Construct a scaled Gaussian, using Stable(2,0,s,m) == Normal(m,s*sqrt(2)).
        new_fn = self._wrap(dist.Normal(fn.loc, scale * (2 ** 0.5)), event_dim)
        return new_fn, obs


class StableReparam(Reparam):
    """
    Auxiliary variable reparameterizer for arbitrary
    :class:`~pyro.distributions.Stable` random variables.

    This is useful in inference of non-symmetric
    :class:`~pyro.distributions.Stable` variables because the
    :meth:`~pyro.distributions.Stable.log_prob` is not implemented.

    This reparameterizes a :class:`~pyro.distributions.Stable` random variable
    as sum of two other stable random variables, one symmetric and the other
    totally skewed (applying Property 2.3.a of [1]). The totally skewed
    variable is sampled as in :class:`LatentStableReparam` , and the symmetric
    variable is decomposed as in :class:`SymmetricStableReparam` .

    [1] V. M. Zolotarev (1986)
        "One-dimensional stable distributions"
    """

    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.Stable) and fn.coords == "S0"

        # Strategy: Let X ~ S0(a,b,s,m) be the stable variable of interest.
        # 1. WLOG scale and shift so s=1 and m=0, additionally shifting to convert
        #    from Zolotarev's S parameterization to Nolan's S0 parameterization.
        # 2. Decompose X = S + T, where
        #    S ~ S(a,0,...,0) is symmetric and
        #    T ~ S(a,sgn(b),...,0) is totally skewed.
        # 3. Decompose S = G * sqrt(Z) via the symmetric strategy, where
        #    Z ~ S(a/2,1,...,0) is totally-skewed and
        #    G ~ Normal(0,1) is Gaussian.
        # 4. Defer the totally-skewed Z and T to the Chambers-Mallows-Stuck
        #    strategy: Z = f(Unif,Exp), T = f(Unif,Exp).
        #
        # To derive the parameters of S and T, we solve the equations
        #
        #   T.stability = a            S.stability = a
        #   T.skew = sgn(b)            S.skew = 0
        #   T.loc = 0                  S.loc = 0
        #
        #   s = (S.scale**a + T.scale**a)**(1/a) = 1       # by step 1.
        #
        #       S.skew * S.scale**a + T.skew * T.scale**a
        #   b = ----------------------------------------- = sgn(b) * T.scale**a
        #                S.scale**a + T.scale**a
        # yielding
        #
        #   T.scale = |b| ** (1/a)     S.scale = (1 - |b|) ** (1/a)

        # Draw parameter-free noise.
        proto = fn.stability
        half_pi = proto.new_tensor(math.pi / 2)
        one = proto.new_ones(proto.shape)
        zu = pyro.sample("{}_z_uniform".format(name),
                         self._wrap(dist.Uniform(-half_pi, half_pi).expand(proto.shape), event_dim))
        ze = pyro.sample("{}_z_exponential".format(name),
                         self._wrap(dist.Exponential(one), event_dim))
        tu = pyro.sample("{}_t_uniform".format(name),
                         self._wrap(dist.Uniform(-half_pi, half_pi).expand(proto.shape), event_dim))
        te = pyro.sample("{}_t_exponential".format(name),
                         self._wrap(dist.Exponential(one), event_dim))

        # Differentiably transform.
        a = fn.stability
        z = _unsafe_standard_stable(a / 2, 1, zu, ze, coords="S")
        t = _standard_stable(a, one, tu, te, coords="S0")
        a_inv = a.reciprocal()
        eps = torch.finfo(a.dtype).eps
        skew_abs = fn.skew.abs().clamp(min=eps, max=1 - eps)
        t_scale = skew_abs.pow(a_inv)
        s_scale = (1 - skew_abs).pow(a_inv)
        shift = _safe_shift(a, fn.skew, t_scale, skew_abs)
        loc = fn.loc + fn.scale * (fn.skew.sign() * t_scale * t + shift)
        scale = fn.scale * s_scale * z.sqrt() * (math.pi / 4 * a).cos().pow(a_inv)
        scale = scale.clamp(min=torch.finfo(scale.dtype).tiny)

        # Construct a scaled Gaussian, using Stable(2,0,s,m) == Normal(m,s*sqrt(2)).
        new_fn = self._wrap(dist.Normal(loc, scale * (2 ** 0.5)), event_dim)
        return new_fn, obs


def _unsafe_shift(a, skew, t_scale):
    # At a=1 the lhs has a root and the rhs has an asymptote.
    return (skew.sign() * t_scale - skew) * (math.pi / 2 * a).tan()


def _safe_shift(a, skew, t_scale, skew_abs):
    radius = 0.005
    hole = 1.0
    with torch.no_grad():
        near_hole = (a - hole).abs() <= radius
    if not near_hole.any():
        return _unsafe_shift(a, skew, t_scale)

    # Avoid the hole at a=1 by interpolating between points on either side.
    a_ = a.unsqueeze(-1).expand(a.shape + (2,)).contiguous()
    with torch.no_grad():
        lb, ub = a_.data.unbind(-1)
        lb[near_hole] = hole - radius
        ub[near_hole] = hole + radius
        # We don't need to backprop through weights, since we've pretended
        # a_ is reparametrized, even though we've clamped some values.
        weights = (a_ - a.unsqueeze(-1)).abs_().mul_(-1 / (2 * radius)).add_(1)
        weights[~near_hole] = 0.5
    skew_ = skew.unsqueeze(-1)
    skew_abs_ = skew_abs.unsqueeze(-1)
    t_scale_ = skew_abs_.pow(a_.reciprocal())
    pairs = _unsafe_shift(a_, skew_, t_scale_)
    return (pairs * weights).sum(-1)
