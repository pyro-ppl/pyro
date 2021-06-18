# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from .messenger import Messenger
from .runtime import effectful


@effectful(type="get_init_messengers")
def _get_init_messengers():
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

        # See https://github.com/pyro-ppl/pyro/issues/2878
        # This is a tricky hack to apply messengers in an order other than the
        # standard _PYRO_STACK order. Our goal is to allow (model, initializer)
        # pairs to be reparametrized as a unit. The problem is that messengers
        # are typically applied in the order
        #
        #    InitMessenger(init_to_value(...))(ReparamMessenger(...)(model))
        #
        # so that original model sites are reparametrized by the time they are
        # seen by init_to_value(). To work around this we allow
        # ReparamMessenger to apply enclosing InitMessengers early, simulating
        # a priority system for messengers (indeed we might consider
        # prioritizing messengers). Note that the enclosing InitMessenger will be
        # called a second time, after ReparamMessenger, but that is ok because
        # InitMessenger does not overwrite values.
        #
        # To get this same logic to work for ConditionMessenger or
        # ReplayMessenger we would need to ensure those messengers can
        # similarly be safely applied twice, with the second application
        # avoiding overwriting the original application.
        for m in _get_init_messengers():
            m._pyro_sample(msg)

        # Pass args_kwargs to the reparam via a side channel.
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
