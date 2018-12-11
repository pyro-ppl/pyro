from __future__ import absolute_import, division, print_function

from collections import Counter

from .reentrant_messenger import ReentrantMessenger

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


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
    """
    def __init__(self, history=1, keep=False):
        assert history >= 0
        self.history = history
        self.keep = keep
        self._iterable = None
        self._pos = -1
        self._stack = []
        super(MarkovMessenger, self).__init__()

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
        return super(MarkovMessenger, self).__enter__()

    def __exit__(self, *args, **kwargs):
        if not self.keep:
            self._stack.pop()
        self._pos -= 1
        return super(MarkovMessenger, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        if msg["done"] or type(msg["fn"]).__name__ == "_Subsample":
            return

        # We use a Counter rather than a set here so that sites can correctly
        # go out of scope when any one of their markov contexts exits.
        # This accounting can be done by users of these fields,
        # e.g. EnumerateMessenger.
        infer = msg["infer"]
        scope = infer.setdefault("_markov_scope", Counter())  # site name -> markov depth
        for pos in range(max(0, self._pos - self.history), self._pos + 1):
            scope.update(self._stack[pos])
        infer["_markov_depth"] = 1 + infer.get("_markov_depth", 0)
        self._stack[self._pos].add(msg["name"])
