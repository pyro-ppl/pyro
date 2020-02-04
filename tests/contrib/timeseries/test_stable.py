# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import pyro.distributions as dist
from pyro.contrib.timeseries.stable import LogStableCoxProcess


@pytest.mark.parametrize("hidden_dim,obs_dim", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("fixed", [False, True], ids=["learned", "fixed"])
def test_log_stable_cox_process_fit(hidden_dim, obs_dim, fixed):
    num_steps = 10
    data = dist.Poisson(10.).sample((num_steps, obs_dim))
    process = LogStableCoxProcess("foo", hidden_dim, obs_dim, max_rate=100)
    if fixed:
        # TODO fix numerical issues with gradient of stability.
        process.model.stability = 1.9
    process.fit(data, num_steps=2)
    noise = process.detect(data)
    assert noise["trans"].shape == (num_steps - 1, hidden_dim)
    assert noise["obs"].shape == (num_steps, obs_dim)
