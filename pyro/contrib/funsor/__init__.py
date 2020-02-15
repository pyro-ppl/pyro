# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pyro.poutine.runtime

from .named_messenger import _DIM_STACK, LocalNamedMessenger


@pyro.poutine.runtime.effectful(type="to_funsor")
def to_funsor(x, output=None, dim_to_name=None):
    if pyro.poutine.runtime.am_i_wrapped() and not dim_to_name:
        dim_to_name = _DIM_STACK.global_frame.dim_to_name.copy()
    import funsor
    return funsor.to_funsor(x, output=output, dim_to_name=dim_to_name)


@pyro.poutine.runtime.effectful(type="to_data")
def to_data(x, name_to_dim=None):
    if pyro.poutine.runtime.am_i_wrapped() and not name_to_dim:
        name_to_dim = _DIM_STACK.global_frame.name_to_dim.copy()
    import funsor
    return funsor.to_data(x, name_to_dim=name_to_dim)


def markov(fn=None, history=1, keep=False):
    """
    Handler for converting to/from funsors consistent with Pyro's positional batch dimensions.

    This function is a piece of syntactic sugar that can be used in a variety of ways:
    - as a context manager
    - as a decorator for recursive functions
    - as an iterator for markov chains

    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their share"
    """
    if fn is None:
        # Used as a decorator with bound args
        return LocalNamedMessenger(history=history, keep=keep)
    if not callable(fn):
        # Used as a generator
        return LocalNamedMessenger(history=history, keep=keep).generator(iterable=fn)
    # Used as a decorator with bound args
    return LocalNamedMessenger(history=history, keep=keep)(fn)


__all__ = [
    "to_data",
    "to_funsor",
    "markov",
]
