# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools

from .messenger import Messenger


class ReentrantMessenger(Messenger):
    def __init__(self):
        self._ref_count = 0
        super().__init__()

    def __call__(self, fn):
        return functools.wraps(fn)(super().__call__(fn))

    def __enter__(self):
        self._ref_count += 1
        if self._ref_count == 1:
            super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._ref_count -= 1
        if self._ref_count == 0:
            super().__exit__(exc_type, exc_value, traceback)
