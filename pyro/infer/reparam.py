import re

from torch.distributions import constraints

import pyro
import pyro.poutine as poutine
from pyro.distributions.reparameterize import LocScaleReparameterizer


def decenter(fn, match=".*"):
    """
    Effect for learnable decentering reparameterization.

    For each matched latent sample site ``x``, this creates a new learnable
    parameter ``x_centered`` and applies a
    :class:`~pyro.distributions.reparameterize.LocScaleReparameterizer` .

    :param str match: A regular expression matching site names. Defaults to all
        latent sample sites.
    """

    def config_fn(site):
        if site["type"] == "sample" and not site["is_observed"]:
            if re.match(match, site["name"]):
                centered = pyro.param(
                    "{}_centered".format(site["name"]),
                    lambda: site["fn"].sample().new_full(site["fn"].batch_shape, 0.5),
                    constraint=constraints.unit_interval,
                    event_dim=0)
                # TODO determine shape_params from site["fn"]
                return {"reparam": LocScaleReparameterizer(centered)}
        return {}

    fn = poutine.infer_config(fn, config_fn)
    fn = poutine.reparam(fn)
    return fn
