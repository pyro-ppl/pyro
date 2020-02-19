# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.forecast import ForecastingModel, backtest
from pyro.contrib.forecast.evaluate import DEFAULT_METRICS


class Model(ForecastingModel):
    def model(self, zero_data, covariates):
        loc = zero_data[..., :1, :]
        scale = pyro.sample("scale", dist.LogNormal(loc, 1).to_event(1))

        with self.time_plate:
            jumps = pyro.sample("jumps", dist.Normal(0, scale).to_event(1))
        prediction = jumps.cumsum(-2)

        noise_dist = dist.Normal(zero_data, 1)
        self.predict(noise_dist, prediction)


WINDOWS = [
    (None, 1, None, 1, 8),
    (None, 4, None, 4, 8),
    (10, 1, None, 3, 5),
    (None, 5, 10, 1, 5),
    (7, 1, 7, 1, 7),
]


@pytest.mark.parametrize("train_window,min_train_window,test_window,min_test_window,stride", WINDOWS)
def test_simple(train_window, min_train_window, test_window, min_test_window, stride):
    duration = 30
    obs_dim = 2
    covariates = torch.zeros(duration, 0)
    data = torch.randn(duration, obs_dim) + 4

    metrics = backtest(data, covariates, Model(),
                       train_window=train_window,
                       test_window=test_window,
                       stride=stride,
                       forecaster_options={"num_steps": 2})

    for name in DEFAULT_METRICS:
        for window in metrics.values():
            assert name in window
            assert 0 < window[name] < math.inf


@pytest.mark.parametrize("train_window,min_train_window,test_window,min_test_window,stride", WINDOWS)
def test_poisson(train_window, min_train_window, test_window, min_test_window, stride):
    duration = 30
    obs_dim = 2
    covariates = torch.zeros(duration, 0)
    rate = torch.randn(duration, obs_dim) + 4
    counts = dist.Poisson(rate).sample()

    # Transform count data to log domain.
    data = counts.log1p()

    def transform(pred, truth):
        pred = dist.Poisson(pred.clamp(min=1e-4).expm1()).sample()
        truth = truth.expm1()
        return pred, truth

    metrics = backtest(data, covariates, Model(),
                       transform=transform,
                       train_window=train_window,
                       test_window=test_window,
                       stride=stride,
                       forecaster_options={"num_steps": 2})

    for name in DEFAULT_METRICS:
        for window in metrics.values():
            assert name in window
            assert 0 < window[name] < math.inf
