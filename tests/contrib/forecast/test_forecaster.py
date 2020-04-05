# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.forecast import Forecaster, ForecastingModel, HMCForecaster
from pyro.infer.autoguide import AutoDelta
from pyro.infer.reparam import LinearHMMReparam, StableReparam
from pyro.optim import Adam


class Model0(ForecastingModel):
    def model(self, zero_data, covariates):
        with pyro.plate_stack("batch", zero_data.shape[:-2], rightmost_dim=-2):
            loc = zero_data[..., :1, 0]
            scale = pyro.sample("scale", dist.LogNormal(loc, 1))

            with self.time_plate:
                jumps = pyro.sample("jumps", dist.Normal(0, scale))
            prediction = jumps.cumsum(-1).unsqueeze(-1) + zero_data

            noise_dist = dist.Laplace(zero_data, 1)
            self.predict(noise_dist, prediction)


class Model1(ForecastingModel):
    def model(self, zero_data, covariates):
        with pyro.plate_stack("batch", zero_data.shape[:-2], rightmost_dim=-2):
            loc = zero_data[..., :1, :]
            scale = pyro.sample("scale", dist.LogNormal(loc, 1).to_event(1))

            with self.time_plate:
                jumps = pyro.sample("jumps", dist.Normal(0, scale).to_event(1))
            prediction = jumps.cumsum(-2)

            noise_dist = dist.Laplace(0, 1)
            self.predict(noise_dist, prediction)


class Model2(ForecastingModel):
    def model(self, zero_data, covariates):
        with pyro.plate_stack("batch", zero_data.shape[:-2], rightmost_dim=-2):
            loc = zero_data[..., :1, :]
            scale = pyro.sample("scale", dist.LogNormal(loc, 1).to_event(1))

            with self.time_plate:
                jumps = pyro.sample("jumps", dist.Normal(0, scale).to_event(1))
            prediction = jumps.cumsum(-2)

            scale_tril = torch.eye(zero_data.size(-1))
            noise_dist = dist.MultivariateNormal(zero_data, scale_tril=scale_tril)
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
            noise_dist = dist.GaussianHMM(
                dist.Normal(0, 1).expand([obs_dim]).to_event(1),
                torch.eye(obs_dim),
                dist.Normal(0, 1).expand([obs_dim]).to_event(1),
                torch.eye(obs_dim),
                dist.Normal(0, 1).expand([obs_dim]).to_event(1),
                duration=duration,
            )
            self.predict(noise_dist, prediction)


class Model4(ForecastingModel):
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
            rep = StableReparam()
            with poutine.reparam(config={"residual": LinearHMMReparam(rep, rep, rep)}):
                self.predict(noise_dist, prediction)


@pytest.mark.parametrize("t_obs", [1, 7])
@pytest.mark.parametrize("t_forecast", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("cov_dim", [0, 1, 6])
@pytest.mark.parametrize("obs_dim", [1, 2])
@pytest.mark.parametrize("dct_gradients", [False, True])
@pytest.mark.parametrize("Model", [Model0, Model1, Model2, Model3, Model4])
@pytest.mark.parametrize("engine", ["svi", "hmc"])
def test_smoke(Model, batch_shape, t_obs, t_forecast, obs_dim, cov_dim, dct_gradients, engine):
    model = Model()
    data = torch.randn(batch_shape + (t_obs, obs_dim))
    covariates = torch.randn(batch_shape + (t_obs + t_forecast, cov_dim))

    if engine == "svi":
        forecaster = Forecaster(model, data, covariates[..., :t_obs, :],
                                num_steps=2, log_every=1, dct_gradients=dct_gradients)
    else:
        if dct_gradients is True:
            pytest.skip("Duplicated test.")
        forecaster = HMCForecaster(model, data, covariates[..., :t_obs, :], max_tree_depth=1,
                                   num_warmup=1, num_samples=1, jit_compile=False)

    num_samples = 5
    samples = forecaster(data, covariates, num_samples)
    assert samples.shape == (num_samples,) + batch_shape + (t_forecast, obs_dim,)
    samples = forecaster(data, covariates, num_samples, batch_size=2)
    assert samples.shape == (num_samples,) + batch_shape + (t_forecast, obs_dim,)


@pytest.mark.parametrize("subsample_aware", [False, True])
def test_svi_custom_smoke(subsample_aware):
    t_obs = 5
    t_forecast = 4
    cov_dim = 3
    obs_dim = 2

    model = Model0()
    data = torch.randn(t_obs, obs_dim)
    covariates = torch.randn(t_obs + t_forecast, cov_dim)
    guide = AutoDelta(model)
    optim = Adam({})

    Forecaster(model, data, covariates[..., :t_obs, :],
               guide=guide, optim=optim, subsample_aware=subsample_aware,
               num_steps=2, log_every=1)


class SubsampleModel3(ForecastingModel):
    def model(self, zero_data, covariates):
        with pyro.plate("batch", len(zero_data), dim=-2):
            zero_data = pyro.subsample(zero_data, event_dim=1)
            covariates = pyro.subsample(covariates, event_dim=1)

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


class SubsampleModel4(ForecastingModel):
    def model(self, zero_data, covariates):
        with pyro.plate("batch", len(zero_data), dim=-2):
            zero_data = pyro.subsample(zero_data, event_dim=1)
            covariates = pyro.subsample(covariates, event_dim=1)

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
            rep = StableReparam()
            with poutine.reparam(config={"residual": LinearHMMReparam(rep, rep, rep)}):
                self.predict(noise_dist, prediction)


@pytest.mark.parametrize("t_obs", [1, 7])
@pytest.mark.parametrize("t_forecast", [1, 3])
@pytest.mark.parametrize("cov_dim", [0, 6])
@pytest.mark.parametrize("obs_dim", [1, 2])
@pytest.mark.parametrize("Model", [SubsampleModel3, SubsampleModel4])
def test_subsample_smoke(Model, t_obs, t_forecast, obs_dim, cov_dim):
    batch_shape = (4,)
    model = Model()
    data = torch.randn(batch_shape + (t_obs, obs_dim))
    covariates = torch.randn(batch_shape + (t_obs + t_forecast, cov_dim))

    def create_plates(zero_data, covariates):
        size = len(zero_data)
        return pyro.plate("batch", size, subsample_size=2, dim=-2)

    forecaster = Forecaster(model, data, covariates[..., :t_obs, :],
                            num_steps=2, log_every=1, create_plates=create_plates)

    num_samples = 5
    samples = forecaster(data, covariates, num_samples)
    assert samples.shape == (num_samples,) + batch_shape + (t_forecast, obs_dim,)
    samples = forecaster(data, covariates, num_samples, batch_size=2)
    assert samples.shape == (num_samples,) + batch_shape + (t_forecast, obs_dim,)
