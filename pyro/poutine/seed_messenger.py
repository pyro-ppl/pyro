# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import TracebackType
from typing import Optional, Type

from pyro.poutine.messenger import Messenger
from pyro.util import get_rng_state, set_rng_seed, set_rng_state


class SeedMessenger(Messenger):
    """
    Handler to set the random number generator to a pre-defined state by setting its
    seed. This is the same as calling :func:`pyro.set_rng_seed` before the
    call to `fn`. This handler has no additional effect on primitive statements on the
    standard Pyro backend, but it might intercept ``pyro.sample`` calls in other
    backends. e.g. the NumPy backend.

    :param fn: a stochastic function (callable containing Pyro primitive calls).
    :param int rng_seed: rng seed.
    """

    def __init__(self, rng_seed: int) -> None:
        assert isinstance(rng_seed, int)
        self.rng_seed = rng_seed
        super().__init__()

    def __enter__(self) -> None:  # type: ignore[override]
        self.old_state = get_rng_state()
        set_rng_seed(self.rng_seed)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        set_rng_state(self.old_state)
