# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
The :mod:`pyro.contrib.autoname` module provides tools for automatically
generating unique, semantically meaningful names for sample sites.
"""
from pyro.contrib.autoname import named
from pyro.contrib.autoname.scoping import scope, name_count


__all__ = [
    "named",
    "scope",
    "name_count",
]
