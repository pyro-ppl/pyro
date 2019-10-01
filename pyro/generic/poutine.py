from contextlib import contextmanager

import pyro
from pyro.poutine import *  # noqa: F401, F403


@contextmanager
def seed(rng=None, fn=None):
    pyro.set_rng_seed(rng)
    yield
