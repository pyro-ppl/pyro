# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.forecast import Forecaster, ForecastingModel, HMCForecaster, backtest
from pyro.contrib.forecast.evaluate import DEFAULT_METRICS
from pyro.util import optional


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
    (None, 10, None, 10, 1),
    (10, 1, None, 3, 5),
    (None, 5, 10, 1, 5),
    (7, 1, 7, 1, 7),
    (14, 1, 7, 1, 1),
]


@pytest.mark.parametrize("train_window,min_train_window,test_window,min_test_window,stride", WINDOWS)
@pytest.mark.parametrize("warm_start", [False, True], ids=["cold", "warm"])
def test_simple(train_window, min_train_window, test_window, min_test_window, stride, warm_start):
    duration = 30
    obs_dim = 2
    covariates = torch.zeros(duration, 0)
    data = torch.randn(duration, obs_dim) + 4
    forecaster_options = {"num_steps": 2, "warm_start": warm_start}

    expect_error = (warm_start and train_window is not None)
    with optional(pytest.raises(ValueError), expect_error):
        windows = backtest(data, covariates, Model,
                           train_window=train_window,
                           min_train_window=min_train_window,
                           test_window=test_window,
                           min_test_window=min_test_window,
                           stride=stride,
                           forecaster_options=forecaster_options)
    if not expect_error:
        assert any(window["t0"] == 0 for window in windows)
        if stride == 1:
            assert any(window["t2"] == duration for window in windows)
        for window in windows:
            assert window["train_walltime"] >= 0
            assert window["test_walltime"] >= 0
            for name in DEFAULT_METRICS:
                assert name in window
                assert 0 < window[name] < math.inf


@pytest.mark.parametrize("train_window,min_train_window,test_window,min_test_window,stride", WINDOWS)
@pytest.mark.parametrize("engine", ["svi", "hmc"])
def test_poisson(train_window, min_train_window, test_window, min_test_window, stride, engine):
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

    if engine == "svi":
        forecaster_fn = Forecaster
        forecaster_options = {"num_steps": 2}
    else:
        forecaster_fn = HMCForecaster
        forecaster_options = {"num_warmup": 1, "num_samples": 1}

    windows = backtest(data, covariates, Model,
                       forecaster_fn=forecaster_fn,
                       transform=transform,
                       train_window=train_window,
                       min_train_window=min_train_window,
                       test_window=test_window,
                       min_test_window=min_test_window,
                       stride=stride,
                       forecaster_options=forecaster_options)

    assert any(window["t0"] == 0 for window in windows)
    if stride == 1:
        assert any(window["t0"] == 0 for window in windows)
        assert any(window["t2"] == duration for window in windows)
    for name in DEFAULT_METRICS:
        for window in windows:
            assert name in window
            assert 0 < window[name] < math.inf


def test_custom_warm_start():
    duration = 30
    obs_dim = 2
    covariates = torch.zeros(duration, 0)
    data = torch.randn(duration, obs_dim) + 4
    min_train_window = 10

    def forecaster_options(t0, t1, t2):
        if t1 == min_train_window:
            return {"num_steps": 2, "warm_start": True}
        else:
            return {"num_steps": 0, "warm_start": True}

    backtest(data, covariates, Model,
             min_train_window=min_train_window,
             test_window=10,
             forecaster_options=forecaster_options)
