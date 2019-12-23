from collections import OrderedDict

import torch

from .messenger import Messenger
from .runtime import apply_stack


class ReparamMessenger(Messenger):
    """
    Reparametrizes each affected sample site into one or more auxiliary sample
    sites followed by a deterministic transformation [1].

    To specify :class:`~pyro.distributions.reparameterize.Reparameterizer` s,
    either pass a ``config`` dict to the constructor, configure
    ``site["infer"]["reparam"] = my_reparameterizer`` for each desired sample
    site, or use :func:`~pyro.poutine.infer_config` .

    See `available reparameterizers <distributions.html#reparameterizers>`_

    .. warning:: Reparameterizers are recursive; take care to avoid infinite
        loops in your ``config`` filters.

    [1] Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)
        "Automatic Reparameterisation of Probabilistic Programs"
        https://arxiv.org/pdf/1906.03028.pdf

    :param dict config: Optional configuration mapping site name to
        :class:`~pyro.distributions.reparameterize.Reparameterizer` object.
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        self.config = config

    def _pyro_sample(self, msg):
        if msg["is_observed"]:
            return None
        if msg["name"] in self.config:
            msg["infer"]["reparam"] = self.config[msg["name"]]
        reparam = msg["infer"].get("reparam")
        if reparam is None:
            return None

        # Create auxiliary sites.
        new_fns = reparam.get_dists(msg["fn"])
        assert isinstance(new_fns, OrderedDict)
        new_values = OrderedDict()
        for name, fn in new_fns.items():
            new_msg = msg.copy()
            new_msg["name"] = "{}_{}".format(msg["name"], name)
            new_msg["fn"] = fn
            new_msg["cond_indep_stack"] = ()
            new_msg["infer"] = new_msg["infer"].copy()
            new_msg["infer"]["reparam"] = None
            apply_stack(new_msg)
            new_values[name] = new_msg["value"]

        # Combine auxiliary values via pyro.deterministic().
        value, new_fn = reparam.get_value_and_fn(msg["fn"], new_values)
        assert isinstance(value, torch.Tensor)
        if getattr(msg["fn"], "_validation_enabled", False):
            # Validate while the original msg["fn"] is known.
            msg["fn"]._validate_sample(value)
        msg["value"] = value
        msg["fn"] = new_fn
        msg["is_observed"] = True
