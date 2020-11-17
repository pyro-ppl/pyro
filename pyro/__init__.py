# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pyro.poutine as poutine
from pyro.logger import log
from pyro.poutine import condition, do, markov
from pyro.primitives import (barrier, clear_param_store, deterministic, enable_validation, factor, get_param_store,
                             iarange, irange, module, param, plate, plate_stack, random_module, sample, subsample,
                             validation_enabled)
from pyro.util import set_rng_seed

# After changing this, run scripts/update_version.py
version_prefix = '1.5.1'

# Get the __version__ string from the auto-generated _version.py file, if exists.
try:
    from pyro._version import __version__
except ImportError:
    __version__ = version_prefix

__all__ = [
    "__version__",
    "barrier",
    "clear_param_store",
    "condition",
    "deterministic",
    "do",
    "enable_validation",
    "factor",
    "get_param_store",
    "iarange",
    "irange",
    "log",
    "markov",
    "module",
    "param",
    "plate",
    "plate",
    "plate_stack",
    "poutine",
    "random_module",
    "sample",
    "set_rng_seed",
    "subsample",
    "validation_enabled",
]
