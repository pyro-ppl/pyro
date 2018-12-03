from __future__ import absolute_import, division, print_function

import functools

from .messenger import Messenger


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
