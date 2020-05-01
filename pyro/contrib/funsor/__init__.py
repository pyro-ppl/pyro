# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro.contrib.funsor.handlers.primitives import to_data, to_funsor  # noqa: F401
from pyro.contrib.funsor.handlers import markov  # noqa: F401
from pyro.contrib.funsor.handlers import plate as _plate


def plate(*args, **kwargs):
    return _plate(None, *args, **kwargs)
