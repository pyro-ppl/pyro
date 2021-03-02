# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
The :mod:`pyro.contrib.autoname` module provides tools for automatically
generating unique, semantically meaningful names for sample sites.
"""
from pyro.contrib.autoname import named
from pyro.contrib.autoname.autoname import autoname, sample
from pyro.contrib.autoname.scoping import name_count, scope

__all__ = [
    "named",
    "scope",
    "name_count",
    "autoname",
    "sample",
]
