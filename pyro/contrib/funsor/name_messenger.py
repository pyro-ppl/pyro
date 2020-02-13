# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter, OrderedDict, namedtuple
from contextlib import ExitStack  # python 3

from pyro.poutine.reentrant_messenger import ReentrantMessenger


# name_to_dim : dict, dim_to_name : dict
StackFrame = namedtuple('StackFrame', ['name_to_dim', 'dim_to_name'])  # TODO use class to signify mutability?


class NameMessenger(ReentrantMessenger):
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

    def _pyro_to_data(self, msg):
        funsor_value, name_to_dim = msg["args"]
        if name_to_dim is None:
            name_to_dim = OrderedDict()
            msg["args"] = (funsor_value, name_to_dim)

        scope = msg.setdefault("scope", Counter())
        for _ in self.history:
            scope.subtract(name for name in funsor_value.inputs if name not in name_to_dim)
        for pos in range(max(0, self._pos - self.history), self._pos + 1):
            frame = self._stack[pos]
            scope.update(funsor_value.inputs).subtract(name_to_dim)
            for name, depth in scope:
                if depth == 0:
                    dim = frame.name_to_dim.setdefault(name, min(frame.dim_to_name) - 1)
                    frame.name_to_dim[name] = dim
                    name_to_dim[name] = dim
                    del scope[name]

    def _pyro_to_funsor(self, msg):
        raw_value, output, dim_to_name = msg["args"]
        if dim_to_name is None:
            dim_to_name = OrderedDict()
            msg["args"] = (raw_value, output, dim_to_name)

        event_dim = len(output.shape)
        batch_dim = len(raw_value.shape) - event_dim

        scope = msg.setdefault("scope", Counter())
        for _ in self.history:
            scope.subtract(dim for dim in range(-batch_dim, 0) if dim not in dim_to_name)
        for pos in range(max(0, self._pos - self.history), self._pos + 1):
            frame = self._stack[pos]
            scope.update(dim for dim in range(-batch_dim, 0)).subtract(dim_to_name)
            for dim, depth in scope:
                if depth == 0:
                    name = frame.dim_to_name[dim]
                    dim_to_name[dim] = (name, domain)
                    del scope[dim]
