Forecasting
===========

The forecasting module contains a framework for experimenting with a restricted
class of time series models and inference algorithms. Models include
hierarchical multivariate heavy-tailed time series of ~1000 time steps and
~1000 separate series. Inference combines variational inference with Gaussian
variable elimination based on the :class:`~pyro.distributions.GaussianHMM`
class. Forecasts are in the form of joint posterior samples at multiple future
time steps.

Forecaster Interface
---------------------
.. automodule:: pyro.contrib.forecast.forecaster
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Evaluation
----------
.. automodule:: pyro.contrib.forecast.evaluate
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
