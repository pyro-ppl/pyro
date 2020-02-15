# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.timeseries.forecast import Forecaster, ForecastingModel


class SimpleForecastingModel(ForecastingModel):
    def get_globals(self, zero_data, covariates):
        loc = torch.zeros(zero_data.size(-1))
        scale = pyro.sample("scale",
                            dist.LogNormal(loc, 1)
                                .to_event(1))
        globals_ = scale
        return globals_

    def get_locals(self, zero_data, covariates, globals_):
        scale = globals_
        jumps = pyro.sample("jumps",
                            dist.Normal(0, scale)
                                .expand(zero_data.shape[-2:])
                                .to_event(1))
        prediction = jumps.cumsum(-2)
        locals_ = None
        return prediction, locals_

    def get_dist(self, zero_data, covariates, globals_, locals_):
        loc = zero_data
        return dist.Laplace(loc, 1).to_event(2)


@pytest.mark.parametrize("t_obs", [1, 2, 3])
@pytest.mark.parametrize("t_forecast", [1, 2, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)])
@pytest.mark.parametrize("cov_dim", [0, 1, 4])
@pytest.mark.parametrize("obs_dim", [1, 2, 3])
@pytest.mark.parametrize("Model", [ForecastingModel, SimpleForecastingModel])
def test_smoke(Model, batch_shape, t_obs, t_forecast, obs_dim, cov_dim):
    model = Model()
    data = torch.randn(batch_shape + (t_obs, obs_dim))
    covariates = torch.randn(batch_shape + (t_obs + t_forecast, cov_dim))

    forecaster = Forecaster(model, data, covariates[..., :t_obs, :],
                            num_steps=2, log_every=1)

    num_samples = 5
    samples = forecaster(data, covariates, num_samples)
    assert samples.shape == (num_samples,) + batch_shape + (t_forecast, obs_dim,)
