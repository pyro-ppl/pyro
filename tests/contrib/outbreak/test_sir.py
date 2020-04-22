# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyro.contrib.outbreak import SIRModel


@pytest.mark.parametrize("duration", [2, 10])
@pytest.mark.parametrize("forecast", [0, 10])
def test_smoke(duration, forecast):
    population = 100
    recovery_time = 7.0

    model = SIRModel(population, recovery_time, [None] * duration)
    data = model.generate()["obs"]

    model = SIRModel(population, recovery_time, data)
    model.fit(warmup_steps=2, num_samples=4, max_tree_depth=2)

    model.predict(forecast=forecast)
