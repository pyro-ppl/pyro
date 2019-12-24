import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist


class CumsumReparam:
    """
    Cumsum reparameterization.

    This is useful when a time series is parameterized by increments, which is
    very poorly conditioned. This changes to a cumsum(increnents)
    parameterization.
    """
    def __call__(self, name, fn, obs):
        assert obs is None
        assert fn.event_dim == 1
        size = fn.event_shape[0]
        scale = pyro.param("{}_scale", torch.tensor(1.),
                           constraint=constraints.positive)
        cumsum_fn = dist.Cauchy(0, scale).expand([size + 1])
        cumsum_value = pyro.sample("{}_cumsum".format(name), cumsum_fn)

        value = cumsum_value[..., 1:] - cumsum_value[..., :-1]
        log_density = fn.log_prob(value) - cumsum_fn.log_prob(cumsum_value)
        new_fn = dist.Delta(value, log_density, event_dim=1)
        return new_fn, value
