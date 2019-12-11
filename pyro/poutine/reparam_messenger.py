from collections import OrderedDict

import torch

from pyro.distributions.delta import Delta

from .messenger import Messenger
from .runtime import apply_stack


class ReparamMessenger(Messenger):
    """
    Reparametrizes each affected sample site into one or more auxiliary sample
    sites followed by a deterministic transformation [1].

    To specify :class:`~pyro.distributions.reparameterize.Reparameterizer` s
    either configure ``site["infer"]["reparam"] = my_reparameterizer`` for each
    desired sample site or use :func:`~pyro.poutine.config_enumerate` .
    Only to sites marked ``site["infer"]["reparam"] = ...`` will be affected.

    See `available reparameterizers <distributions.html#reparameterizers>`_

    .. warning:: Reparameterizers are recursive; take care to avoid infinite
        loops in your ``@infer_config`` filters.

    [1] Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)
        "Automatic Reparameterisation of Probabilistic Programs"
        https://arxiv.org/pdf/1906.03028.pdf
    """
    def _pyro_sample(self, msg):
        if msg["is_observed"]:
            return None
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
        # TODO(https://github.com/pyro-ppl/pyro/issues/2214) refactor to
        # use site type "deterministic" when it exists.
        value = reparam.transform_values(msg["fn"], new_values)
        assert isinstance(value, torch.Tensor)
        if getattr(msg["fn"], "_validation_enabled", False):
            # Validate while the original msg["fn"] is known.
            msg["fn"]._validate_sample(value)
        msg["value"] = value
        msg["fn"] = Delta(value, event_dim=msg["fn"].event_dim).mask(False)
        msg["is_observed"] = True
