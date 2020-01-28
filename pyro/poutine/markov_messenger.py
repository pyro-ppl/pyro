# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from contextlib import ExitStack  # python 3

from .reentrant_messenger import ReentrantMessenger


class MarkovMessenger(ReentrantMessenger):
    """
    Markov dependency declaration.

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
            self._stack.append(set())
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if not self.keep:
            self._stack.pop()
        self._pos -= 1
        return super().__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        if msg["done"] or type(msg["fn"]).__name__ == "_Subsample":
            return

        # We use a Counter rather than a set here so that sites can correctly
        # go out of scope when any one of their markov contexts exits.
        # This accounting can be done by users of these fields,
        # e.g. EnumMessenger.
        infer = msg["infer"]
        scope = infer.setdefault("_markov_scope", Counter())  # site name -> markov depth
        for pos in range(max(0, self._pos - self.history), self._pos + 1):
            scope.update(self._stack[pos])
        infer["_markov_depth"] = 1 + infer.get("_markov_depth", 0)
        self._stack[self._pos].add(msg["name"])
