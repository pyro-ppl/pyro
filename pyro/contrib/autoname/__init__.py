"""
The :mod:`pyro.contrib.autoname` module provides tools for automatically
generating unique, semantically meaningful names for sample sites.
"""
from __future__ import absolute_import, division, print_function

from pyro.contrib.autoname import named
from pyro.contrib.autoname.scoping import scope


__all__ = [
    "named",
    "scope"
]
