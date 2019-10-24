"""
The :mod:`pyro.contrib.timeseries` module provides a collection of Bayesian time series
models useful for forecasting applications.
"""
from pyro.contrib.timeseries.gp import IndependentMaternGP, CoupledMaternGP

__all__ = [
    "CoupledMaternGP",
    "IndependentMaternGP",
]
