# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Callable, Optional, Union

from .messenger import Messenger

_STRATEGIES = {}


def register_reparam_strategy(name: str, fn: Optional[Callable] = None):
    """
    Registers a named reparametrization strategy.

    Example::

        @register_reparam_strategy("custom")
        def my_strategy(msg):
            ...

        with poutine.reparam(config="custom"):
            ...

    :param str name: Name of the strategy.
    :param callable fn: The function. If missing, this returns a decorator.
    """
    assert isinstance(name, str)
    if fn is None:
        return functools.partial(register_reparam_strategy, name)
    assert callable(fn)
    _STRATEGIES[name] = fn


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
    def __init__(self, config: Union[dict, Callable, str]):
        super().__init__()
        if isinstance(config, str):
            config = _STRATEGIES[config]
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

        reparam.args_kwargs = self._args_kwargs
        try:
            new_fn, value = reparam(msg["name"], msg["fn"], msg["value"])
        finally:
            reparam.args_kwargs = None

        if value is not None:
            if msg["value"] is None:
                msg["is_observed"] = True
            msg["value"] = value
            if getattr(msg["fn"], "_validation_enabled", False):
                # Validate while the original msg["fn"] is known.
                msg["fn"]._validate_sample(value)
        msg["fn"] = new_fn


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
