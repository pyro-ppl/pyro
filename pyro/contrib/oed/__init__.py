"""
The :mod:`pyro.contrib.oed` module provides tools to create optimal experiment
designs for pyro models. In particular, it provides estimators for the
average posterior entropy (APE) criterion.

To estimate the APE for a particular design, use::

    def model(design):
        ...

    eig = vi_ape(model, design, ...)

APE can then be minimised using existing optimisers in :mod:`pyro.optim`.
"""

from pyro.contrib.oed import search, eig

__all__ = [
    "search",
    "eig"
]
