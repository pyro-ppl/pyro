# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from .messenger import Messenger
from .runtime import effectful


@effectful(type="get_init_messengers")
def get_init_messengers():
    return []


class ReparamMessenger(Messenger):
    """
    Reparametrizes each affected sample site into one or more auxiliary sample
    sites followed by a deterministic transformation [1].

    To specify reparameterizers, pass a ``config`` dict or callable to the
    constructor.  See the :mod:`pyro.infer.reparam` module for available
    reparameterizers.

    Note some reparameterizers can examine the ``*args,**kwargs`` inputs of
    functions they affect; these reparameterizers require using
    ``poutine.reparam`` as a decorator rather than as a context manager.

    [1] Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)
        "Automatic Reparameterisation of Probabilistic Programs"
        https://arxiv.org/pdf/1906.03028.pdf

    :param config: Configuration, either a dict mapping site name to
        :class:`~pyro.infer.reparam.reparam.Reparameterizer` ,
        or a function mapping site to
        :class:`~pyro.infer.reparam.reparam.Reparameterizer` or None.
    :type config: dict or callable
    """
    def __init__(self, config):
        super().__init__()
        assert isinstance(config, dict) or callable(config)
        self.config = config
        self._args_kwargs = None

    def __call__(self, fn):
        return ReparamHandler(self, fn)

    def _pyro_sample(self, msg):
        if isinstance(self.config, dict):
            reparam = self.config.get(msg["name"])
        else:
            reparam = self.config(msg)
        if reparam is None:
            return

        # Apply init messengers at higher priority than reparam.
        for m in get_init_messengers():
            m._pyro_sample(msg)

        reparam.args_kwargs = self._args_kwargs
        try:
            new_msg = reparam.apply({
                "name": msg["name"],
                "fn": msg["fn"],
                "value": msg["value"],
                "is_observed": msg["is_observed"],
            })
        finally:
            reparam.args_kwargs = None

        msg["is_observed"] = new_msg["is_observed"]
        if msg["value"] is None and new_msg["value"] is not None:
            # Simulate a pyro.deterministic() site.
            msg["value"] = new_msg["value"]
            assert msg["is_observed"]

            # Validate while the original msg["fn"] is known.
            if getattr(msg["fn"], "_validation_enabled", False):
                msg["fn"]._validate_sample(msg["value"])
        msg["fn"] = new_msg["fn"]


class ReparamHandler(object):
    """
    Reparameterization poutine.
    """
    def __init__(self, msngr, fn):
        self.msngr = msngr
        self.fn = fn

    def __call__(self, *args, **kwargs):
        # This saves args,kwargs for optional use by reparameterizers.
        self.msngr._args_kwargs = args, kwargs
        try:
            with self.msngr:
                return self.fn(*args, **kwargs)
        finally:
            self.msngr._args_kwargs = None
