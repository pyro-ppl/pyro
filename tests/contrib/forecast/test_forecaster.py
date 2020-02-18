# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.forecast import Forecaster, ForecastingModel
from pyro.infer.reparam import LinearHMMReparam, StableReparam


class Model1(ForecastingModel):
    def model(self, zero_data, covariates):
        with pyro.plate_stack("batch", zero_data.shape[:-2], rightmost_dim=-2):
            loc = zero_data[..., :1, :]
            scale = pyro.sample("scale", dist.LogNormal(loc, 1).to_event(1))

            with self.time_plate:
                jumps = pyro.sample("jumps", dist.Normal(0, scale).to_event(1))
            prediction = jumps.cumsum(-2)

            noise_dist = dist.Laplace(zero_data, 1).to_event(2)
            self.predict(noise_dist, prediction)


class Model2(ForecastingModel):
    def model(self, zero_data, covariates):
        with pyro.plate_stack("batch", zero_data.shape[:-2], rightmost_dim=-2):
            loc = zero_data[..., :1, :]
            scale = pyro.sample("scale", dist.LogNormal(loc, 1).to_event(1))

            with self.time_plate:
                jumps = pyro.sample("jumps", dist.Normal(0, scale).to_event(1))
            prediction = jumps.cumsum(-2)

            duration, obs_dim = zero_data.shape[-2:]
            noise_dist = dist.GaussianHMM(
                dist.Normal(0, 1).expand([obs_dim]).to_event(1),
                torch.eye(obs_dim),
                dist.Normal(0, 1).expand([obs_dim]).to_event(1),
                torch.eye(obs_dim),
                dist.Normal(0, 1).expand([obs_dim]).to_event(1),
                duration=duration,
            )
            self.predict(noise_dist, prediction)


class Model3(ForecastingModel):
    def model(self, zero_data, covariates):
        with pyro.plate_stack("batch", zero_data.shape[:-2], rightmost_dim=-2):
            loc = zero_data[..., :1, :]
            scale = pyro.sample("scale", dist.LogNormal(loc, 1).to_event(1))

            with self.time_plate:
                jumps = pyro.sample("jumps", dist.Normal(0, scale).to_event(1))
            prediction = jumps.cumsum(-2)

            duration, obs_dim = zero_data.shape[-2:]
            noise_dist = dist.LinearHMM(
                dist.Stable(1.9, 0).expand([obs_dim]).to_event(1),
                torch.eye(obs_dim),
                dist.Stable(1.9, 0).expand([obs_dim]).to_event(1),
                torch.eye(obs_dim),
                dist.Stable(1.9, 0).expand([obs_dim]).to_event(1),
                duration=duration,
            )
            with poutine.reparam(config={
                    "residual": LinearHMMReparam(StableReparam(), StableReparam(), StableReparam())}):
                self.predict(noise_dist, prediction)


@pytest.mark.parametrize("t_obs", [1, 7])
@pytest.mark.parametrize("t_forecast", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("cov_dim", [0, 1, 6])
@pytest.mark.parametrize("obs_dim", [1, 2])
@pytest.mark.parametrize("Model", [Model1, Model2, Model3])
def test_smoke(Model, batch_shape, t_obs, t_forecast, obs_dim, cov_dim):
    model = Model()
    data = torch.randn(batch_shape + (t_obs, obs_dim))
    covariates = torch.randn(batch_shape + (t_obs + t_forecast, cov_dim))

    forecaster = Forecaster(model, data, covariates[..., :t_obs, :],
                            num_steps=2, log_every=1)

    num_samples = 5
    samples = forecaster(data, covariates, num_samples)
    assert samples.shape == (num_samples,) + batch_shape + (t_forecast, obs_dim,)
