from __future__ import absolute_import, division, print_function

import functools
from collections import Counter

from .messenger import Messenger

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


class ReentrantMessenger(Messenger):
    def __init__(self):
        self._ref_count = 0
        super(ReentrantMessenger, self).__init__()

    def __call__(self, fn):
        return functools.wraps(fn)(super(ReentrantMessenger, self).__call__(fn))

    def __enter__(self):
        self._ref_count += 1
        if self._ref_count == 1:
            super(ReentrantMessenger, self).__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._ref_count -= 1
        if self._ref_count == 0:
            super(ReentrantMessenger, self).__exit__(exc_type, exc_value, traceback)


class MarkovMessenger(ReentrantMessenger):
    """
    Markov dependency declaration.

    This is a statistical equivalent of a memory management arena.

    .. warning:: This assumes markov contexts can be reentrant but cannot be
        interleaved. I.e. the following is invalid::

        with pyro.markov() as x_axis:
            with pyro.markov():
                with x_axis:  # <--- error here
                    ...

    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their shared ancestors).
    """
    def __init__(self, history=1, keep=False):
        assert history >= 0
        self.history = history
        self.keep = keep
        self._iterable = None

        self._pos = -1
        self._stack = []

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
            self._stack.push(set())
        return super(MarkovMessenger, self).__enter__()

    def __exit__(self, *args, **kwargs):
        if not self.keep:
            self._stack.pop()
        self._pos -= 1
        # FIXME handle exceptions correctly
        return super(MarkovMessenger, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        if msg["done"] or msg["is_observed"]:
            return

        infer = msg["infer"]
        infer["_markov_depth"] = 1 + infer.get("_markov_depth", 0)
        upstream = infer.setdefault("_markov_upstream", Counter())
        for pos in range(max(0, self._pos - self.history), self._pos + 1):
            upstream.update(self._stack[pos])
