from __future__ import absolute_import, division, print_function

import functools

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
    Statistical equivalent of a memory management arena.

    .. warning:: This assumes markov contexts can be reentrant but cannot be
        interleaved. I.e. the following is invalid::

        with pyro.markov() as x_axis:
            with pyro.markov():
                with x_axis:  # <--- error here
                    ...
    """
    def __init__(self, history=1, keep=False):
        assert history > 0  # otherwise we need more precise slicing below
        self.history = history
        self.keep = keep
        self.iterator = None

        self._pos = -1
        self._stack = []
        self._set = set()  # set of active dimensions

        # _ref_counts is shared among all MarkovMessengers, with lifetime of a model call.
        self._ref_counts = None

    def generator(self, iterable):
        self.iterable = list(iterable)
        return self

    def __iter__(self):
        with ExitStack() as stack:
            for value in self.iterable:
                stack.enter_context(self)
                yield value

    def __enter__(self):
        self._pos += 1
        if len(self._stack) <= self._pos:
            self._stack.push(set())
        self._set = set().union(*self._stack[min(0, self._pos - self.history):1 + self._pos])
        return super(MarkovMessenger, self).__enter__()

    def __exit__(self, *args, **kwargs):
        if not self.keep:
            self._stack.pop()
        self._pos -= 1
        self._set = set().union(*self._stack[min(0, self._pos - self.history):1 + self._pos])
        if self._pos == -1:
            self._ref_counts = None
        # FIXME handle exceptions correctly
        return super(MarkovMessenger, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        if "cond_dep_set" in msg:
            msg["cond_dep_set"] &= self._set
        else:
            msg["cond_dep_set"] = self._set.copy()

        self._markov_depth = msg["infer"].get("_markov_depth", defaultdict(int))
        self._markov_depth[msg["name"]] += 1

    def _pyro_post_sample(self, msg):
        if self._markov_depths[msg["name"]]
        msg["infer"]["_markov_depth"] -= 1
        if msg["infer"]["_markov_depth"] == 0:
            dim = msg["infer"].get("_enumerate_dim")
            if dim is not None:
                self._stack[self._pos].add(dim)
                self._set.add(dim)
