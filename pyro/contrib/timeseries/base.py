# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pyro.nn import PyroModule, pyro_method


class TimeSeriesModel(PyroModule):
    """
    Base class for univariate and multivariate time series models.
    """
    @pyro_method
    def log_prob(self, targets):
        """
        Log probability function.

        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape ``(T, obs_dim)``, where ``T`` is the length of the time series and ``obs_dim``
            is the dimension of the real-valued ``targets`` at each time step
        :returns torch.Tensor: A 0-dimensional log probability for the case of properly
            multivariate time series models in which the output dimensions are correlated;
            otherwise returns a 1-dimensional tensor of log probabilities for batched
            univariate time series models.
        """
        raise NotImplementedError

    @pyro_method
    def forecast(self, targets, dts):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape ``(T, obs_dim)``, where ``T`` is the length of the time series and ``obs_dim``
            is the dimension of the real-valued targets at each time step. These
            represent the training data that are conditioned on for the purpose of making
            forecasts.
        :param torch.Tensor dts: A 1-dimensional tensor of times to forecast into the future,
            with zero corresponding to the time of the final target ``targets[-1]``.
        :returns torch.distributions.Distribution: Returns a predictive distribution with batch shape ``(S,)`` and
            event shape ``(obs_dim,)``, where ``S`` is the size of ``dts``. That is, the resulting
            predictive distributions do not encode correlations between distinct times in ``dts``.
        """
        raise NotImplementedError

    def get_dist(self):
        """
        Get a :class:`~pyro.distributions.Distribution` object corresponding to
        this time series model.  Often this is a
        :class:`~pyro.distributions.GaussianHMM`.
        """
        raise NotImplementedError
