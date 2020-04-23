# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyro.contrib.outbreak import SIRModel


@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("forecast", [0, 7])
def test_smoke(duration, forecast):
    population = 100
    recovery_time = 7.0

    # Generate data.
    model = SIRModel(population, recovery_time, [None] * duration)
    for attempt in range(100):
        data = model.generate({"R0": 1.5, "rho": 0.5})["obs"]
        if data.sum():
            break
    assert data.sum() > 0, "failed to generate positive data"

    # Infer.
    model = SIRModel(population, recovery_time, data)
    num_samples = 5
    model.fit(warmup_steps=2, num_samples=num_samples, max_tree_depth=2)

    # Predict and forecast.
    samples = model.predict(forecast=forecast)
    assert samples["S"].shape == (num_samples, duration + forecast)
