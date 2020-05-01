# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest
import torch

from pyro.contrib.epidemiology import OverdispersedSIRModel, SimpleSIRModel, SparseSIRModel

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
@pytest.mark.parametrize("options", [
    {},
    {"dct": 1.},
    {"num_quant_bins": 8},
    {"num_quant_bins": 12},
    {"num_quant_bins": 16},
], ids=str)
def test_simple_smoke(duration, forecast, options):
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
    {"dct": 1.},
    {"num_quant_bins": 8},
], ids=str)
def test_overdispersed_smoke(duration, forecast, options):
    population = 100
    recovery_time = 7.0

    # Generate data.
    model = OverdispersedSIRModel(population, recovery_time, [None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5, "k": 1.0})["obs"]
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


@pytest.mark.parametrize("duration", [4, 12])
@pytest.mark.parametrize("forecast", [7])
@pytest.mark.parametrize("options", [
    {},
    {"dct": 1.},
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
