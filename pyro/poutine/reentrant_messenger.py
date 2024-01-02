# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
from types import TracebackType
from typing import Callable, Optional, Type, TypeVar

from typing_extensions import ParamSpec, Self

from pyro.poutine.messenger import Messenger

P = ParamSpec("P")
T = TypeVar("T")


class ReentrantMessenger(Messenger):
    def __init__(self) -> None:
        self._ref_count = 0
        super().__init__()

    def __call__(self, fn: Callable[P, T]) -> Callable[P, T]:
        return functools.wraps(fn)(super().__call__(fn))

    def __enter__(self) -> Self:
        self._ref_count += 1
        if self._ref_count == 1:
            super().__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_value: Optional[Exception],
        traceback: Optional[TracebackType],
    ) -> None:
        self._ref_count -= 1
        if self._ref_count == 0:
            super().__exit__(exc_type, exc_value, traceback)
