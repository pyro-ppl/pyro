"""
The :mod:`pyro.contrib.oed` module provides tools to create optimal experiment
designs for pyro models. In particular, it provides estimators for the
expected information gain (EIG) criterion.

To estimate the EIG for a particular design, use::

    def model(design):
        ...

    # Select an appropriate EIG estimator, such as
    eig = vnmc_eig(model, design, ...)

EIG can then be maximised using existing optimisers in :mod:`pyro.optim`.
"""

from pyro.contrib.oed import search, eig

__all__ = [
    "search",
    "eig"
]
