# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple
from contextlib import ExitStack  # python 3

from pyro.poutine.reentrant_messenger import ReentrantMessenger


# name_to_dim : dict, dim_to_name : dict
StackFrame = namedtuple('StackFrame', ['name_to_dim', 'dim_to_name'])


class NamedMessenger(ReentrantMessenger):
    """
    This is a statistical equivalent of a memory management arena.

    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their shared ancestors).
    :param int dim: An optional dimension to use for this independence index.
        Interface stub, behavior not yet implemented.
    :param str name: An optional unique name to help inference algorithms match
        :func:`pyro.markov` sites between models and guides.
        Interface stub, behavior not yet implemented.
    """
    def __init__(self, history=1, keep=False, dim=None, name=None):
        assert history >= 0
        self.history = history
        self.keep = keep
        self.dim = dim
        self.name = name
        if dim is not None:
            raise NotImplementedError(
                "vectorized markov not yet implemented, try setting dim to None")
        if name is not None:
            raise NotImplementedError(
                "vectorized markov not yet implemented, try setting name to None")
        self._iterable = None
        self._pos = -1
        self._stack = []
        super().__init__()

    def generator(self, iterable):
        self._iterable = iterable
        return self

    def __iter__(self):
        with ExitStack() as stack:
            for value in self._iterable:
                stack.enter_context(self)
                yield value

    def __enter__(self):
        self._pos += 1
        if len(self._stack) <= self._pos:
            self._stack.append(
                StackFrame(name_to_dim=OrderedDict(), dim_to_name=OrderedDict()))
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if not self.keep:
            self._stack.pop()
        self._pos -= 1
        return super().__exit__(*args, **kwargs)

    @property
    def current_frame(self):
        return self._stack[self._pos]

    def _pyro_to_data(self, msg):
        if len(msg["args"]) < 2:
            msg["args"] = msg["args"] + (OrderedDict(),)
        funsor_value, name_to_dim = msg["args"]

        # every markov context is responsible for knowing about non-fresh local vars,
        # and fresh vars are handled by passing information up & down the stack

        # handling non-fresh vars
        name_to_dim.update({name: self.current_frame.name_to_dim[name]
                            for name in funsor_value.inputs
                            if name in self.current_frame.name_to_dim
                            and name not in name_to_dim})

        # handling fresh vars
        depth = msg.setdefault("depth", -self.history - 1)
        if depth == -self.history - 1:
            msg["fresh"] = frozenset(name for name in funsor_value.inputs
                                     if name not in name_to_dim
                                     and name not in self.current_frame.name_to_dim)
        fresh = msg["fresh"]

        for pos in reversed(range(max(0, self._pos - self.history), self._pos + 1)):
            frame = self._stack[pos]
            msg["depth"] += 1
            if msg["depth"] == 0:
                for name in fresh:
                    if name not in frame.name_to_dim:
                        dim = -1
                        while dim in frame.dim_to_name:
                            dim -= 1
                        frame.name_to_dim[name] = dim
                    dim = frame.name_to_dim[name]
                    frame.dim_to_name[dim] = (name, funsor_value.inputs[name])
                    name_to_dim[name] = dim
        msg['depth'] += self._pos

    def _pyro_post_to_data(self, msg):
        funsor_value, name_to_dim = msg["args"]

        msg['depth'] -= self._pos

        # copy fresh variables down the stack
        for pos in range(max(0, self._pos - self.history), self._pos + 1):
            frame = self._stack[pos]
            if msg["depth"] <= 0 and msg["depth"] >= -self.history - 1:
                frame.name_to_dim.update({name: name_to_dim[name] for name in msg["fresh"]})
                # dim_to_name maps dims to (name, domain) pairs
                frame.dim_to_name.update(
                    {name_to_dim[name]: (name, funsor_value.inputs[name]) for name in msg["fresh"]})

            msg["depth"] -= 1

    def _pyro_to_funsor(self, msg):
        if len(msg["args"]) < 3:
            msg["args"] = msg["args"] + (OrderedDict(),)
        raw_value, output, dim_to_name = msg["args"]

        event_dim = len(output.shape)
        batch_dim = len(raw_value.shape) - event_dim

        # since we don't allow fresh positional dims, only look at current frame
        dim_to_name.update({dim: self.current_frame.dim_to_name[dim]
                            for dim in range(-batch_dim, 0)
                            if dim not in dim_to_name
                            and dim in self.current_frame.dim_to_name})
