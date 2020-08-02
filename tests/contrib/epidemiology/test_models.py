# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest
import torch

import pyro.distributions as dist
from pyro.contrib.epidemiology.models import (HeterogeneousRegionalSIRModel, HeterogeneousSIRModel,
                                              OverdispersedSEIRModel, OverdispersedSIRModel, RegionalSIRModel,
                                              SimpleSEIRDModel, SimpleSEIRModel, SimpleSIRModel, SparseSIRModel,
                                              SuperspreadingSEIRModel, SuperspreadingSIRModel, UnknownStartSIRModel)
from tests.common import xfail_param

logger = logging.getLogger(__name__)


@pytest.mark.filterwarnings("ignore:num_chains")
@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("algo,options", [
    ("svi", {}),
    ("svi", {"haar": False}),
    ("svi", {"guide_rank": None}),
    ("svi", {"guide_rank": 2}),
    ("svi", {"guide_rank": "full"}),
    ("mcmc", {}),
    ("mcmc", {"haar": False}),
    ("mcmc", {"haar_full_mass": 0}),
    ("mcmc", {"haar_full_mass": 2}),
    ("mcmc", {"num_quant_bins": 2}),
    ("mcmc", {"num_quant_bins": 4}),
    ("mcmc", {"num_quant_bins": 8}),
    ("mcmc", {"num_quant_bins": 12}),
    ("mcmc", {"num_quant_bins": 16}),
    ("mcmc", {"num_quant_bins": 2, "haar": False}),
    ("mcmc", {"arrowhead_mass": True}),
    ("mcmc", {"jit_compile": True}),
    ("mcmc", {"jit_compile": True, "haar_full_mass": 0}),
    ("mcmc", {"jit_compile": True, "num_quant_bins": 2}),
    ("mcmc", {"num_chains": 2, "mp_context": "spawn"}),
    ("mcmc", {"num_chains": 2, "mp_context": "spawn", "num_quant_bins": 2}),
    ("mcmc", {"num_chains": 2, "mp_context": "spawn", "jit_compile": True}),
], ids=str)
def test_simple_sir_smoke(duration, forecast, options, algo):
    population = 100
    recovery_time = 7.0

    # Generate data.
    model = SimpleSIRModel(population, recovery_time, [None] * duration)
    assert model.full_mass == [("R0", "rho")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SimpleSIRModel(population, recovery_time, data)
    num_samples = 5
    if algo == "mcmc":
        model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)
    else:
        model.fit_svi(num_steps=2, num_samples=num_samples, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    num_samples *= options.get("num_chains", 1)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("algo,options", [
    ("svi", {}),
    ("svi", {"haar": False}),
    ("mcmc", {}),
    ("mcmc", {"haar": False}),
    ("mcmc", {"haar_full_mass": 0}),
    ("mcmc", {"num_quant_bins": 2}),
], ids=str)
def test_simple_seir_smoke(duration, forecast, options, algo):
    population = 100
    incubation_time = 2.0
    recovery_time = 7.0

    # Generate data.
    model = SimpleSEIRModel(population, incubation_time, recovery_time,
                            [None] * duration)
    assert model.full_mass == [("R0", "rho")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SimpleSEIRModel(population, incubation_time, recovery_time, data)
    num_samples = 5
    if algo == "mcmc":
        model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)
    else:
        model.fit_svi(num_steps=2, num_samples=num_samples, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["E"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("algo,options", [
    ("svi", {}),
    ("mcmc", {}),
    ("mcmc", {"haar_full_mass": 0}),
], ids=str)
def test_simple_seird_smoke(duration, forecast, options, algo):
    population = 100
    incubation_time = 2.0
    recovery_time = 7.0
    mortality_rate = 0.1

    # Generate data.
    model = SimpleSEIRDModel(population, incubation_time, recovery_time,
                             mortality_rate, [None] * duration)
    assert model.full_mass == [("R0", "rho")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SimpleSEIRDModel(population, incubation_time, recovery_time,
                             mortality_rate, data)
    num_samples = 5
    if algo == "mcmc":
        model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)
    else:
        model.fit_svi(num_steps=2, num_samples=num_samples, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["E"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)
    assert samples["D"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3])
@pytest.mark.parametrize("forecast", [7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": False},
    {"num_quant_bins": 2},
], ids=str)
def test_overdispersed_sir_smoke(duration, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    model = OverdispersedSIRModel(population, recovery_time, [None] * duration)
    assert model.full_mass == [("R0", "rho", "od")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = OverdispersedSIRModel(population, recovery_time, data)
    num_samples = 5
    model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3])
@pytest.mark.parametrize("forecast", [7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": False},
    {"num_quant_bins": 2},
], ids=str)
def test_overdispersed_seir_smoke(duration, forecast, options):
    population = 100
    incubation_time = 2.0
    recovery_time = 7.0

    # Generate data.
    model = OverdispersedSEIRModel(population, incubation_time, recovery_time,
                                   [None] * duration)
    assert model.full_mass == [("R0", "rho", "od")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = OverdispersedSEIRModel(population, incubation_time, recovery_time, data)
    num_samples = 5
    model.fit_mcmc(warmup_steps=2, num_samples=num_samples, max_tree_depth=2,
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
    {"haar": False},
    {"haar_full_mass": 0},
    {"num_quant_bins": 2},
], ids=str)
def test_superspreading_sir_smoke(duration, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    model = SuperspreadingSIRModel(population, recovery_time, [None] * duration)
    assert model.full_mass == [("R0", "k", "rho")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5, "k": 1.0})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SuperspreadingSIRModel(population, recovery_time, data)
    num_samples = 5
    model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("options", [
    {},
    {"haar": False},
    {"haar_full_mass": 0},
    {"num_quant_bins": 2},
], ids=str)
def test_superspreading_seir_smoke(duration, forecast, options):
    population = 100
    incubation_time = 2.0
    recovery_time = 7.0

    # Generate data.
    model = SuperspreadingSEIRModel(
        population, incubation_time, recovery_time, [None] * duration)
    assert model.full_mass == [("R0", "k", "rho")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5, "k": 1.0})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SuperspreadingSEIRModel(
        population, incubation_time, recovery_time, data)
    num_samples = 5
    model.fit_mcmc(warmup_steps=2, num_samples=num_samples, max_tree_depth=2,
                   **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["E"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("algo,options", [
    ("svi", {}),
    ("svi", {"haar": False}),
    ("mcmc", {}),
    ("mcmc", {"num_quant_bins": 2}),
], ids=str)
def test_coalescent_likelihood_smoke(duration, forecast, options, algo):
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
    if algo == "mcmc":
        model.fit_mcmc(warmup_steps=2, num_samples=num_samples, max_tree_depth=2,
                       **options)
    else:
        model.fit_svi(num_steps=2, num_samples=num_samples, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["E"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("algo,options", [
    ("svi", {}),
    ("svi", {"haar": False}),
    ("mcmc", {}),
    ("mcmc", {"haar": False}),
    ("mcmc", {"num_quant_bins": 2}),
], ids=str)
def test_heterogeneous_sir_smoke(duration, forecast, options, algo):
    population = 100
    recovery_time = 7.0

    # Generate data.
    model = HeterogeneousSIRModel(population, recovery_time, [None] * duration)
    assert model.full_mass == [("R0", "rho0", "rho1", "rho2")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = HeterogeneousSIRModel(population, recovery_time, data)
    num_samples = 5
    model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
    assert samples["I"].shape == (num_samples, duration + forecast)
    assert samples["beta"].shape == (num_samples, duration + forecast)


@pytest.mark.parametrize("duration", [4, 12])
@pytest.mark.parametrize("forecast", [7])
@pytest.mark.parametrize("options", [
    xfail_param({}, reason="Delta is incompatible with relaxed inference"),
    {"num_quant_bins": 2},
    {"num_quant_bins": 2, "haar": False},
    {"num_quant_bins": 2, "haar_full_mass": 0},
    {"num_quant_bins": 4},
], ids=str)
def test_sparse_smoke(duration, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    data = [None] * duration
    mask = torch.arange(duration) % 4 == 3
    model = SparseSIRModel(population, recovery_time, data, mask)
    assert model.full_mass == [("R0", "rho")]
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
    model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

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
    {"haar": False},
    {"haar_full_mass": 0},
    {"num_quant_bins": 2},
], ids=str)
def test_unknown_start_smoke(duration, pre_obs_window, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    data = [None] * duration
    model = UnknownStartSIRModel(population, recovery_time, pre_obs_window, data)
    assert model.full_mass == [("R0", "rho0", "rho1")]
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
    model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)

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
@pytest.mark.parametrize("algo,options", [
    ("svi", {}),
    ("svi", {"haar": False}),
    ("mcmc", {}),
    ("mcmc", {"haar": False}),
    ("mcmc", {"haar_full_mass": 0}),
    ("mcmc", {"num_quant_bins": 2}),
], ids=str)
def test_regional_smoke(duration, forecast, options, algo):
    num_regions = 6
    coupling = torch.eye(num_regions).clamp(min=0.1)
    population = torch.tensor([2., 3., 4., 10., 100., 1000.])
    recovery_time = 7.0

    # Generate data.
    model = RegionalSIRModel(population, coupling, recovery_time,
                             data=[None] * duration)
    assert model.full_mass == [("R0", "rho_c1", "rho_c0", "rho")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        assert data.shape == (duration, num_regions)
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = RegionalSIRModel(population, coupling, recovery_time, data)
    num_samples = 5
    if algo == "mcmc":
        model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)
    else:
        model.fit_svi(num_steps=2, num_samples=num_samples, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast, num_regions)
    assert samples["I"].shape == (num_samples, duration + forecast, num_regions)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("algo,options", [
    ("svi", {}),
    ("svi", {"haar": False}),
    ("mcmc", {}),
    ("mcmc", {"haar": False}),
    ("mcmc", {"haar_full_mass": 0}),
    ("mcmc", {"num_quant_bins": 2}),
    ("mcmc", {"jit_compile": True}),
    ("mcmc", {"jit_compile": True, "haar": False}),
    ("mcmc", {"jit_compile": True, "num_quant_bins": 2}),
], ids=str)
def test_hetero_regional_smoke(duration, forecast, options, algo):
    num_regions = 6
    coupling = torch.eye(num_regions).clamp(min=0.1)
    population = torch.tensor([2., 3., 4., 10., 100., 1000.])
    recovery_time = 7.0

    # Generate data.
    model = HeterogeneousRegionalSIRModel(population, coupling, recovery_time,
                                          data=[None] * duration)
    assert model.full_mass == [("R0", "R_drift", "rho0", "rho_drift")]
    for attempt in range(100):
        data = model.generate({"R0": 1.5})["obs"]
        assert data.shape == (duration, num_regions)
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = HeterogeneousRegionalSIRModel(population, coupling, recovery_time, data)
    num_samples = 5
    if algo == "mcmc":
        model.fit_mcmc(warmup_steps=1, num_samples=num_samples, max_tree_depth=2, **options)
    else:
        model.fit_svi(num_steps=2, num_samples=num_samples, **options)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast, num_regions)
    assert samples["I"].shape == (num_samples, duration + forecast, num_regions)
