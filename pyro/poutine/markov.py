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
        if self.ref_count == 1:
            super(ReentrantMessenger, self).__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._ref_count -= 1
        if self._ref_count == 0:
            super(ReentrantMessenger, self).__exit__(exc_type, exc_value, traceback)


class MarkovMessenger(ReentrantMessenger):
    """
    Statistical equivalent of an memory management arena.

    .. warning:: This assumes markov contexts can be reentrant but cannot be
        interleaved. I.e. the following is invalid::

        with pyro.markov() as x_axis:
            with pyro.markov():
                with x_axis:  # <--- error here
                    ...
    """
    def __init__(self, window=1):
        assert window > 0  # otherwise we need more precise slicing below
        self.window = window
        self._stack = []  # list of sets of dimensions
        self._set = set()  # set of active dimensions
        self.iterator = None

    def generator(self, iterable):
        self.iterable = list(iterable)
        return self

    def __iter__(self):
        with ExitStack() as stack:
            for value in self.iterable:
                stack.enter_context(self)
                yield value

    def __enter__(self):
        self._stack.push(set())
        self._set = set().union(*self._stack[-self.window:])
        return super(MarkovMessenger, self).__enter__()

    def __exit__(self, *args, **kwargs):
        self._stack.pop()
        self._set = set().union(*self._stack[-self.window:])
        # FIXME handle exceptions correctly
        return super(MarkovMessenger, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        cond_dep_set = msg.setdefault("cond_dep_set", self._set)
        if cond_dep_set is not self._set:
            msg["cond_dep_set"] = cond_dep_set & self._set
        if "_markov" in msg["infer"]:
            msg["infer"]["_markov"] += 1
        else:
            msg["infer"]["_markov"] = 0

    def _pyro_post_sample(self, msg):
        msg["infer"]["_markov"] -= 1
        if msg["infer"]["_markov"] == 0:
            dim = msg["infer"].get("_enumerate_dim")
            if dim is not None:
                self._stack[dim].add(dim)
                self._set.add(dim)
