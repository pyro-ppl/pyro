import pyro
import pyro.distributions as dist
from pyro.ops.tensor_utils import inverse_cumsum


class CumsumReparam:
    """
    Cumsum reparameterization.

    This is useful when a time series is parameterized by increments: the
    posterior is often poorly conditioned in this representation but
    well-conditioned on the cumsum of increments. This changes to the
    cumsum(increments) parameterization.

    The following are equivalent models, but (2) and (3) have better geometry.

    1. Naive parameterization in terms of increments::

        z = pyro.sample("z", my_increment_dist)
        z_cumsum = z.cumsum()
        # ...observe statements involving z_cumsum...

    2. Manual reparameterization as a factor graph::

        z_cumsum = pyro.sample("z_cumsum",
                               dist.Normal(0, 1)
                                   .expand([size])
                                   .to_event(1)
                                   .mask(False))
        z = z_cumsum[1:] - z_cumsum[:-1]
        value = inverse_cumsum(z_cumsum, dim=-1)
        pyro.sample("z", my_increment_dist, obs=z)
        # ...observe statements involving z_cumsum...

    3. Automatic reparameterization using pyro.reparam::

        z = pyro.sample("z", my_increment_dist)
        z_cumsum = z.cumsum()
        # ...observe statements involving z_cumsum...


    This reparameterization works only for latent variables, not likelihoods.
    """
    def __call__(self, name, fn, obs):
        assert fn.event_dim == 1
        assert obs is None
        value_cumsum = pyro.sample("{}_cumsum".format(name),
                                   dist.Normal(0, 1)
                                       .expand(fn.event_shape)
                                       .to_event(1)
                                       .mask(False))

        value = inverse_cumsum(value_cumsum, dim=-1)
        new_fn = dist.Delta(value, log_density=fn.log_prob(value), event_dim=1)
        return new_fn, value
