# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest
import torch

import pyro.distributions as dist
from pyro.contrib.epidemiology import (OverdispersedSEIRModel, OverdispersedSIRModel, RegionalSIRModel, SimpleSEIRModel,
                                       SimpleSIRModel, SparseSIRModel, SuperspreadingSEIRModel, SuperspreadingSIRModel,
                                       UnknownStartSIRModel)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": True},
    {"haar_full_mass": 2},
    {"num_quant_bins": 2},
    {"num_quant_bins": 8},
    {"num_quant_bins": 12},
    {"num_quant_bins": 16},
    {"arrowhead_mass": True},
], ids=str)
def test_simple_sir_smoke(duration, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    model = SimpleSIRModel(population, recovery_time, [None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SimpleSIRModel(population, recovery_time, data)
    num_samples = 5
    model.fit(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": True},
    {"haar_full_mass": 2},
    {"num_quant_bins": 8},
], ids=str)
def test_simple_seir_smoke(duration, forecast, options):
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


@pytest.mark.parametrize("duration", [3])
@pytest.mark.parametrize("forecast", [7])
@pytest.mark.parametrize("options", [{}, {"haar_full_mass": 2}], ids=str)
def test_overdispersed_sir_smoke(duration, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    model = OverdispersedSIRModel(population, recovery_time, [None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = OverdispersedSIRModel(population, recovery_time, data)
    num_samples = 5
    model.fit(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3])
@pytest.mark.parametrize("forecast", [7])
@pytest.mark.parametrize("options", [{}, {"haar_full_mass": 2}], ids=str)
def test_overdispersed_seir_smoke(duration, forecast, options):
    population = 100
    incubation_time = 2.0
    recovery_time = 7.0

    # Generate data.
    model = OverdispersedSEIRModel(population, incubation_time, recovery_time,
                                   [None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = OverdispersedSEIRModel(population, incubation_time, recovery_time, data)
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
def test_superspreading_sir_smoke(duration, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    model = SuperspreadingSIRModel(population, recovery_time, [None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5, "k": 1.0})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SuperspreadingSIRModel(population, recovery_time, data)
    num_samples = 5
    model.fit(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": True},
    {"haar_full_mass": 2},
    {"num_quant_bins": 8},
], ids=str)
def test_superspreading_seir_smoke(duration, forecast, options):
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


@pytest.mark.parametrize("duration", [4, 12])
@pytest.mark.parametrize("forecast", [7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": True},
    {"haar_full_mass": 3},
    {"num_quant_bins": 8},
], ids=str)
def test_sparse_smoke(duration, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    data = [None] * duration
    mask = torch.arange(duration) % 4 == 3
    model = SparseSIRModel(population, recovery_time, data, mask)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"
    assert (data[1:] >= data[:-1]).all()
    data[~mask] = math.nan
    logger.info("data:\n{}".format(data))

    # Infer.
    model = SparseSIRModel(population, recovery_time, data, mask)
    num_samples = 5
    model.fit(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)
    assert samples["O"].shape == (num_samples, duration + forecast)
    assert (samples["O"][..., 1:] >= samples["O"][..., :-1]).all()
    for O in samples["O"]:
        logger.info("imputed:\n{}".format(O))
        assert (O[:duration][mask] == data[mask]).all()


@pytest.mark.parametrize("pre_obs_window", [6])
@pytest.mark.parametrize("duration", [8])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": True},
    {"haar_full_mass": 4},
    {"num_quant_bins": 8},
], ids=str)
def test_unknown_start_smoke(duration, pre_obs_window, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    data = [None] * duration
    model = UnknownStartSIRModel(population, recovery_time, pre_obs_window, data)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho0": 0.1, "rho1": 0.5})["obs"]
        assert len(data) == pre_obs_window + duration
        data = data[pre_obs_window:]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"
    logger.info("data:\n{}".format(data))

    # Infer.
    model = UnknownStartSIRModel(population, recovery_time, pre_obs_window, data)
    num_samples = 5
    model.fit(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, pre_obs_window + duration + forecast)
    assert samples["I"].shape == (num_samples, pre_obs_window + duration + forecast)

    # Check time of first infection.
    t = samples["first_infection"]
    logger.info("first_infection:\n{}".format(t))
    assert t.shape == (num_samples,)
    assert (0 <= t).all()
    assert (t < pre_obs_window + duration).all()
    for I, ti in zip(samples["I"], t):
        assert (I[:ti] == 0).all()
        assert I[ti] > 0


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": True},
    {"haar_full_mass": 2},
    {"num_quant_bins": 8},
], ids=str)
def test_regional_smoke(duration, forecast, options):
    num_regions = 6
    coupling = torch.eye(num_regions).clamp(min=0.1)
    population = torch.tensor([2., 3., 4., 10., 100., 1000.])
    recovery_time = 7.0

    # Generate data.
    model = RegionalSIRModel(population, coupling, recovery_time,
                             data=[None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        assert data.shape == (duration, num_regions)
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = RegionalSIRModel(population, coupling, recovery_time, data)
    num_samples = 5
    model.fit(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast, num_regions)
    assert samples["I"].shape == (num_samples, duration + forecast, num_regions)
