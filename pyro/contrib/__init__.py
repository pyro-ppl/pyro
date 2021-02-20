# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

r"""
Contributed Code
================

.. warning:: Code in ``pyro.contrib`` is under various stages of development.
    This code makes no guarantee about maintaining backwards compatibility.
"""

from pyro.contrib import autoname, bnn, easyguide, epidemiology, forecast, gp, oed, tracking

__all__ = [
    "autoname",
    "bnn",
    "easyguide",
    "epidemiology",
    "forecast",
    "gp",
    "oed",
    "tracking",
]


try:
    import funsor as funsor_  # noqa: F401
    from pyro.contrib import funsor
    __all__ += ["funsor"]
except ImportError:
    pass
