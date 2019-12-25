import pyro.distributions as dist

from .messenger import Messenger


class ReparamMessenger(Messenger):
    """
    Reparametrizes each affected sample site into one or more auxiliary sample
    sites followed by a deterministic transformation [1].

    To specify reparameterizers, either pass a ``config`` dict to the
    constructor, configure ``site["infer"]["reparam"] = my_reparameterizer``
    for each desired sample site, or use :func:`~pyro.poutine.infer_config` .

    See the :mod:`pyro.infer.reparam` module for available reparameterizers.

    .. warning:: Reparameterizers are recursive; take care to avoid infinite
        loops in your ``config`` filters.

    [1] Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)
        "Automatic Reparameterisation of Probabilistic Programs"
        https://arxiv.org/pdf/1906.03028.pdf

    :param dict config: Optional configuration mapping site name to
        reparameterizer.
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        self.config = config

    def _pyro_sample(self, msg):
        if msg["name"] in self.config:
            msg["infer"]["reparam"] = self.config[msg["name"]]
        reparam = msg["infer"].get("reparam")
        if reparam is None:
            return
        if isinstance(msg["fn"], dist.Delta):
            return  # avoid infinite recursion

        new_fn, value = reparam(msg["name"], msg["fn"], msg["value"])
        if value is not None:
            if msg["value"] is None:
                msg["is_observed"] = True
            msg["value"] = value
            if getattr(msg["fn"], "_validation_enabled", False):
                # Validate while the original msg["fn"] is known.
                msg["fn"]._validate_sample(value)
        msg["fn"] = new_fn
