"""
The :mod:`pyro.contrib.oed` module provides tools to create optimal experiment
designs for pyro models. In particular, it provides estimators for the
expeceted information gain (EIG) criterion.

To estimate the EIG for a particular design, use::

    def model(design):
        ...

    eig = vi_ape(model, design, ...)

EIG can then be optimised using existing optimisers in :mod:`pyro.optim`.
"""

from pyro.contrib.oed import search, eig

__all__ = [
    "search",
    "eig"
]
