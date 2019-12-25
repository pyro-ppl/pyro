import math

import pyro
import pyro.distributions as dist
from pyro.distributions.stable import _standard_stable


class StableReparam:
    """
    Auxiliary variable reparameterizer for
    :class:`~pyro.distributions.Stable` distributions.

    This is useful in inference of latent :class:`~pyro.distributions.Stable`
    variables because the :meth:`~pyro.distributions.Stable.log_prob` is not
    implemented.  This creates a pair of parameter-free auxiliary distributions
    (``Uniform(-pi/2,pi/2)`` and ``Exponential(1)``) with well-defined
    ``.log_prob()`` methods, thereby permitting use of reparameterized stable
    distributions in likelihood-based inference algorithms like SVI and MCMC.

    This reparameterization works only for latent variables, not likelihoods.
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
