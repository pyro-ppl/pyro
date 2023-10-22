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
from .plate_messenger import VectorizedMarkovMessenger
from .replay_messenger import ReplayMessenger
from .trace_messenger import TraceMessenger


@_make_handler(EnumMessenger)
def enum(fn=None, first_available_dim=None):
    ...


enum.__module__ = __name__


@_make_handler(MarkovMessenger)
def markov(fn=None, history=1, keep=False):
    ...


markov.__module__ = __name__


@_make_handler(NamedMessenger)
def named(fn=None, first_available_dim=None):
    ...


named.__module__ = __name__


@_make_handler(ReplayMessenger)
def replay(fn=None, trace=None, params=None):
    ...


replay.__module__ = __name__


@_make_handler(TraceMessenger)
def trace(fn=None, graph_type=None, param_only=None, pack_online=True):
    ...


trace.__module__ = __name__


@_make_handler(VectorizedMarkovMessenger)
def vectorized_markov(fn=None, name=None, size=None, dim=None, history=1):
    ...


vectorized_markov.__module__ = __name__
