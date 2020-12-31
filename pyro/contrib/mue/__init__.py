# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
"""
The :mod:`pyro.contrib.mue` module provides tools for working with mutational
emission (MuE) distributions.
See Weinstein and Marks (2020),
https://www.biorxiv.org/content/10.1101/2020.07.31.231381v1.
Primary developer is Eli N. Weinstein (https://eweinstein.github.io/).
"""
from pyro.contrib.mue.statearrangers import profile
from pyro.contrib.mue.variablelengthhmm import VariableLengthDiscreteHMM

__all__ = [
    "profile"
    "VariableLengthDiscreteHMM"
]
