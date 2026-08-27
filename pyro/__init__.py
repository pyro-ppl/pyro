# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from importlib.metadata import PackageNotFoundError, version

import pyro.poutine as poutine
from pyro.infer.inspect import render_model
from pyro.logger import log
from pyro.poutine import condition, do, markov
from pyro.primitives import (
    barrier,
    clear_param_store,
    deterministic,
    enable_validation,
    factor,
    get_param_store,
    iarange,
    irange,
    module,
    param,
    plate,
    plate_stack,
    random_module,
    sample,
    subsample,
    validation_enabled,
)
from pyro.util import set_rng_seed

from . import settings


def _get_version():
    pyproject_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")
    try:
        with open(pyproject_path, encoding="utf-8") as f:
            match = re.search(r'^version = "([^"]+)"', f.read(), re.MULTILINE)
    except OSError:
        match = None
    if match:
        return match.group(1)

    try:
        return version("pyro-ppl")
    except PackageNotFoundError:
        pass
    return "0.0.0"


__version__ = _get_version()

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
    "render_model",
    "sample",
    "set_rng_seed",
    "settings",
    "subsample",
    "validation_enabled",
]
