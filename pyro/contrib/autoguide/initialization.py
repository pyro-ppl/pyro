r"""
The pyro.contrib.autoguide module contains initialization functions for
automatic guides.

The standard interface for initialization is a function that inputs a Pyro
trace ``site`` dict and returns an appropriately sized ``value`` to serve
as an initial constrained value for a guide estimate.
"""
from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import transform_to

from pyro.poutine.messenger import Messenger
from pyro.util import torch_isnan


def init_to_feasible(site):
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    value = site["fn"].sample().detach()
    t = transform_to(site["fn"].support)
    return t(torch.zeros_like(t.inv(value)))


def init_to_sample(site):
    """
    Initialize to a random sample from the prior.
    """
    return site["fn"].sample().detach()


def init_to_median(site, num_samples=15):
    """
    Initialize to the prior median; fallback to a feasible point if median is
    undefined.
    """
    try:
        # Try to compute empirical median.
        samples = site["fn"].sample(sample_shape=(num_samples,))
        value = samples.median(dim=0)[0]
        if torch_isnan(value):
            raise ValueError
        if hasattr(site["fn"], "_validate_sample"):
            site["fn"]._validate_sample(value)
        return value
    except (RuntimeError, ValueError):
        # Fall back to feasible point.
        return init_to_feasible(site)


def init_to_mean(site):
    """
    Initialize to the prior mean; fallback to median if mean is undefined.
    """
    try:
        # Try .mean() method.
        value = site["fn"].mean.detach()
        if torch_isnan(value):
            raise ValueError
        if hasattr(site["fn"], "_validate_sample"):
            site["fn"]._validate_sample(value)
        return value
    except (NotImplementedError, ValueError):
        # Fall back to a median.
        # This is requred for distributions with infinite variance, e.g. Cauchy.
        return init_to_median(site)


class InitMessenger(Messenger):
    """
    Initializes a site by replacing ``.sample()`` calls with values
    drawn from an initialization strategy. This is mainly for internal use by
    autoguide classes.

    :param callable init_fn: An initialization function.
    """
    def __init__(self, init_fn):
        self.init_fn = init_fn
        super(InitMessenger, self).__init__()

    def _pyro_sample(self, msg):
        if msg["done"] or msg["is_observed"] or type(msg["fn"]).__name__ == "_Subsample":
            return
        with torch.no_grad():
            msg["value"] = self.init_fn(msg)
        msg["done"] = True
