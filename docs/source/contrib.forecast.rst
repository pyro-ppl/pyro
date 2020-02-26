Forecasting
===========
.. automodule:: pyro.contrib.forecast

``pyro.contrib.forecast`` is a lightweight framework for experimenting with a
restricted class of time series models and inference algorithms using familiar
Pyro modeling syntax and PyTorch neural networks.

Models include hierarchical multivariate heavy-tailed time series of ~1000 time
steps and ~1000 separate series. Inference combines subsample-compatible
variational inference with Gaussian variable elimination based on the
:class:`~pyro.distributions.GaussianHMM` class. Forecasts are in the form of
joint posterior samples at multiple future time steps.

Hierarchical models use the familiar :class:`~pyro.plate` syntax for
general hierarchical modeling in Pyro. Plates can be subsampled, enabling
training of joint models over thousands of time series. Multivariate
observations are handled via multivariate likelihoods like
:class:`~pyro.distributions.MultivariateNormal`, :class:`~pyro.distributions.GaussianHMM`, or
:class:`~pyro.distributions.LinearHMM`. Heavy tailed models are possible by
using :class:`~pyro.distributions.StudentT` or
:class:`~pyro.distributions.Stable` likelihoods, possibly together with
:class:`~pyro.distributions.LinearHMM` and reparameterizers including
:class:`~pyro.infer.reparam.studentt.StudentTReparam`,
:class:`~pyro.infer.reparam.stable.StableReparam`, and
:class:`~pyro.infer.reparam.hmm.LinearHMMReparam`.

See :mod:`pyro.contrib.timeseries` for ways to construct temporal Gaussian processes useful as likelihoods.

See the `forecasting example <http://pyro.ai/examples/forecasting_simple.html>`_ for example usage. 

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
