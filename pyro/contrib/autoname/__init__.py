"""
The :mod:`pyro.contrib.autoname` module provides tools for automatically
generating unique, semantically meaningful names for sample sites.
"""
from __future__ import absolute_import, division, print_function

from pyro.contrib.autoname import named
from pyro.contrib.autoname.glom_named import glom_name


__all__ = [
    "glom_name",
    "named",
]
