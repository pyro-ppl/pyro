# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro.poutine import (  # noqa: F401
    block,
    condition,
    do,
    escape,
    infer_config,
    mask,
    reparam,
    scale,
    seed,
    uncondition,
)
from pyro.poutine.handlers import _make_handler

from .enum_messenger import EnumMessenger, queue  # noqa: F401
from .named_messenger import MarkovMessenger, NamedMessenger
from .plate_messenger import PlateMessenger, VectorizedMarkovMessenger
from .replay_messenger import ReplayMessenger
from .trace_messenger import TraceMessenger

_msngrs = [
    EnumMessenger,
    MarkovMessenger,
    NamedMessenger,
    PlateMessenger,
    ReplayMessenger,
    TraceMessenger,
    VectorizedMarkovMessenger,
]

for _msngr_cls in _msngrs:
    _handler_name, _handler = _make_handler(_msngr_cls)
    _handler.__module__ = __name__
    locals()[_handler_name] = _handler
