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


@_make_handler(EnumMessenger, __name__)
def enum(fn=None, first_available_dim=None): ...


@_make_handler(MarkovMessenger, __name__)
def markov(fn=None, history=1, keep=False): ...


@_make_handler(NamedMessenger, __name__)
def named(fn=None, first_available_dim=None): ...


@_make_handler(PlateMessenger, __name__)
def plate(
    fn=None,
    name=None,
    size=None,
    subsample_size=None,
    subsample=None,
    dim=None,
    use_cuda=None,
    device=None,
): ...


@_make_handler(ReplayMessenger, __name__)
def replay(fn=None, trace=None, params=None): ...


@_make_handler(TraceMessenger, __name__)
def trace(fn=None, graph_type=None, param_only=None, pack_online=True): ...


@_make_handler(VectorizedMarkovMessenger, __name__)
def vectorized_markov(fn=None, name=None, size=None, dim=None, history=1): ...
