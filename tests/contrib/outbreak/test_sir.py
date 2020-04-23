# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyro.contrib.outbreak import SIRModel


@pytest.mark.parametrize("duration", [2, 10])
@pytest.mark.parametrize("forecast", [0, 10])
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
    model.fit(warmup_steps=2, num_samples=4, max_tree_depth=2)

    # Predict and forecast.
    model.predict(forecast=forecast)
