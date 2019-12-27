import math

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.stable import _standard_stable, _unsafe_standard_stable
from pyro.infer.util import is_validation_enabled

from .reparam import Reparam


class StableReparam(Reparam):
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
    :class:`SymmetricStableReparam` .

    [1] J.P. Nolan (2017).
        Stable Distributions: Models for Heavy Tailed Data.
        http://fs2.american.edu/jpnolan/www/stable/chap1.pdf
    """
    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.Stable)
        assert obs is None, "StableReparam does not support observe statements"

        # Draw parameter-free noise.
        proto = fn.stability
        half_pi = proto.new_full(proto.shape, math.pi / 2)
        one = proto.new_ones(proto.shape)
        u = pyro.sample("{}_uniform".format(name),
                        self._wrap(dist.Uniform(-half_pi, half_pi), event_dim))
        e = pyro.sample("{}_exponential".format(name),
                        self._wrap(dist.Exponential(one), event_dim))

        # Differentiably transform.
        x = _standard_stable(fn.stability, fn.skew, u, e)
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
        https://people.maths.ox.ac.uk/~howison/papers/levy.pdf
    """
    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.Stable)
        if is_validation_enabled():
            if not (fn.skew == 0).all():
                raise ValueError("SymmetricStableReparam found nonzero skew")
            if not (fn.stability < 2).all():
                raise ValueError("SymmetricStableReparam found stability >= 2")

        # Draw parameter-free noise.
        proto = fn.stability
        half_pi = proto.new_full(proto.shape, math.pi / 2)
        one = proto.new_ones(proto.shape)
        u = pyro.sample("{}_uniform".format(name),
                        self._wrap(dist.Uniform(-half_pi, half_pi), event_dim))
        e = pyro.sample("{}_exponential".format(name),
                        self._wrap(dist.Exponential(one), event_dim))

        # Differentiably transform to scale drawn from a totally-skewed stable variable.
        _, z = _unsafe_standard_stable(fn.stability / 2, 1, u, e)
        assert (z >= 0).all()
        scale = fn.scale * z * (math.pi / 4 * fn.stability).cos().pow(2 / fn.stability)
        scale = scale.clamp(min=torch.finfo(scale.dtype).tiny)

        # Construct a scaled Gaussian.
        new_fn = dist.Normal(fn.loc, scale)
        return new_fn, obs
