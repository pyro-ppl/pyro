# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyroapi

from pyro.contrib.funsor.handlers import condition, do, markov, vectorized_markov
from pyro.contrib.funsor.handlers import plate as _plate
from pyro.contrib.funsor.handlers.primitives import to_data, to_funsor
from pyro.primitives import (
    clear_param_store,
    deterministic,
    enable_validation,
    factor,
    get_param_store,
    module,
    param,
    random_module,
    sample,
    set_rng_seed,
    subsample,
)


def plate(*args, **kwargs):
    return _plate(None, *args, **kwargs)


pyroapi.register_backend(
    "contrib.funsor",
    {
        "distributions": "pyro.distributions",
        "handlers": "pyro.contrib.funsor.handlers",
        "infer": "pyro.contrib.funsor.infer",
        "ops": "torch",
        "optim": "pyro.optim",
        "pyro": "pyro.contrib.funsor",
    },
)

__all__ = [
    "clear_param_store",
    "condition",
    "deterministic",
    "do",
    "enable_validation",
    "factor",
    "get_param_store",
    "markov",
    "module",
    "param",
    "random_module",
    "sample",
    "set_rng_seed",
    "subsample",
    "to_data",
    "to_funsor",
    "vectorized_markov",
]
