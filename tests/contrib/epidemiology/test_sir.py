# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyro.contrib.epidemiology import SimpleSIRModel


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
