from __future__ import absolute_import, division, print_function

from pyro.logger import log
import pyro.poutine as poutine
from pyro.poutine import condition, do
from pyro.primitives import (clear_param_store, enable_validation, get_param_store, iarange, irange, module, param,
                             random_module, sample, validation_enabled)
from pyro.util import set_rng_seed

version_prefix = '0.2.1'

# Get the __version__ string from the auto-generated _version.py file, if exists.
try:
    from pyro._version import __version__
except ImportError:
    __version__ = version_prefix

__all__ = [
    "__version__",
    "clear_param_store",
    "condition",
    "do",
    "enable_validation",
    "get_param_store",
    "iarange",
    "irange",
    "log",
    "module",
    "param",
    "poutine",
    "random_module",
    "sample",
    "set_rng_seed",
    "validation_enabled",
]
