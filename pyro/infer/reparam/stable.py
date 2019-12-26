import math

import pyro
import pyro.distributions as dist
from pyro.distributions.stable import _standard_stable
from pyro.infer.util import is_validation_enabled


class StableReparam:
    """
    Auxiliary variable reparameterizer for
    :class:`~pyro.distributions.Stable` distributions.

    This is useful in inference of latent :class:`~pyro.distributions.Stable`
    variables because the :meth:`~pyro.distributions.Stable.log_prob` is not
    implemented.

    This creates a pair of parameter-free auxiliary distributions
    (``Uniform(-pi/2,pi/2)`` and ``Exponential(1)``) with well-defined
    ``.log_prob()`` methods, thereby permitting use of reparameterized stable
    distributions in likelihood-based inference algorithms like SVI and MCMC.

    This reparameterization works only for latent variables, not likelihoods.
    For likelihoods see :class:`SymmetricStableReparam` .
    """
    def __call__(self, name, fn, obs):
        assert isinstance(fn, dist.Stable)
        assert obs is None, "stable_reparam does not support observe statements"

        # Draw parameter-free noise.
        proto = fn.stability
        half_pi = proto.new_full(proto.shape, math.pi / 2)
        one = proto.new_ones(proto.shape)
        u = pyro.sample("{}_uniform".format(name),
                        dist.Uniform(-half_pi, half_pi))
        e = pyro.sample("{}_exponential".format(name),
                        dist.Exponential(one))

        # Differentiably transform.
        x = _standard_stable(fn.stability, fn.skew, u, e)
        value = fn.loc + fn.scale * x

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=fn.event_dim).mask(False)
        return new_fn, value


class SymmetricStableReparam:
    """
    Auxiliary variable reparameterizer for symmetric
    :class:`~pyro.distributions.Stable` distributions (i.e. those for which
    ``skew=0``).

    This is useful in inference of symmetric
    :class:`~pyro.distributions.Stable` variables because the
    :meth:`~pyro.distributions.Stable.log_prob` is not implemented.

    This reparameterizes a symmetric :class:`~pyro.distributions.Stable` random
    variable as a totally-skewed (``skew=1``)
    :class:`~pyro.distributions.Stable` mixture of
    :class:`~pyro.distributions.Normal` random variables. See Proposition 3. of
    [1].

    [1] Alvaro Cartea and Sam Howison (2009)
        "Option Pricing with Levy-Stable Processes"
        https://people.maths.ox.ac.uk/~howison/papers/levy.pdf
    """
    def __call__(self, name, fn, obs):
        assert isinstance(fn, dist.Stable)
        if is_validation_enabled():
            if not (fn.skew == 0).all():
                raise ValueError("SymmetricStableReparam found nonzero skew")

        # Draw a scale from a totally-skewed stable variable.
        skewed_scale = (math.pi / 4 * fn.stability).cos().pow(2 / fn.stability)
        scale = pyro.sample("{}_scale".format(name),
                            dist.Stable(fn.stability, 1, skewed_scale, 0),
                            infer={"reparam": StableReparam()})

        # Construct a scaled Gaussian.
        new_fn = dist.Normal(fn.loc, fn.scale * scale)
        return new_fn, obs
