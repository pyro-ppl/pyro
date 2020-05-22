# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from pyro.contrib.epidemiology import SimpleSEIRModel, SuperspreadingSEIRModel


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": True},
    {"haar_full_mass": 2},
    {"num_quant_bins": 8},
], ids=str)
def test_simple_smoke(duration, forecast, options):
    population = 100
    incubation_time = 2.0
    recovery_time = 7.0

    # Generate data.
    model = SimpleSEIRModel(population, incubation_time, recovery_time,
                            [None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SimpleSEIRModel(population, incubation_time, recovery_time, data)
    num_samples = 5
    model.fit(warmup_steps=2, num_samples=num_samples, max_tree_depth=2,
              **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["E"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": True},
    {"haar_full_mass": 2},
    {"num_quant_bins": 8},
], ids=str)
def test_overdispersed_smoke(duration, forecast, options):
    population = 100
    incubation_time = 2.0
    recovery_time = 7.0

    # Generate data.
    model = SuperspreadingSEIRModel(
        population, incubation_time, recovery_time, [None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5, "k": 1.0})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SuperspreadingSEIRModel(
        population, incubation_time, recovery_time, data)
    num_samples = 5
    model.fit(warmup_steps=2, num_samples=num_samples, max_tree_depth=2,
              **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["E"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
def test_coalescent_likelihood_smoke(duration, forecast):
    population = 100
    incubation_time = 2.0
    recovery_time = 7.0

    # Generate data.
    model = SuperspreadingSEIRModel(
        population, incubation_time, recovery_time, [None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5, "k": 1.0})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"
    leaf_times = torch.rand(5).pow(0.5) * duration
    coal_times = dist.CoalescentTimes(leaf_times).sample()
    coal_times = coal_times[..., torch.randperm(coal_times.size(-1))]

    # Infer.
    model = SuperspreadingSEIRModel(
        population, incubation_time, recovery_time, data,
        leaf_times=leaf_times, coal_times=coal_times)
    num_samples = 5
    model.fit(warmup_steps=2, num_samples=num_samples, max_tree_depth=2)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["E"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)
