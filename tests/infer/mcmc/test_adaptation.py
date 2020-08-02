# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.infer.mcmc.adaptation import (
    ArrowheadMassMatrix,
    BlockMassMatrix,
    WarmupAdapter,
    adapt_window,
)
from tests.common import assert_close, assert_equal


@pytest.mark.parametrize("adapt_step_size, adapt_mass, warmup_steps, expected", [
    (False, False, 100, []),
    (False, True, 50, [(0, 6), (7, 44), (45, 49)]),
    (True, False, 150, [(0, 74), (75, 99), (100, 149)]),
    (True, True, 200, [(0, 74), (75, 99), (100, 149), (150, 199)]),
    (True, True, 280, [(0, 74), (75, 99), (100, 229), (230, 279)]),
    (True, True, 18, [(0, 17)]),
])
def test_adaptation_schedule(adapt_step_size, adapt_mass, warmup_steps, expected):
    adapter = WarmupAdapter(0.1,
                            adapt_step_size=adapt_step_size,
                            adapt_mass_matrix=adapt_mass)
    adapter.configure(warmup_steps, mass_matrix_shape={"z": (5, 5)})
    expected_schedule = [adapt_window(i, j) for i, j in expected]
    assert_equal(adapter.adaptation_schedule, expected_schedule, prec=0)


@pytest.mark.parametrize("diagonal", [True, False])
def test_arrowhead_mass_matrix(diagonal):
    shape = (2, 3)
    num_samples = 1000

    size = shape[0] * shape[1]
    block_adapter = BlockMassMatrix()
    arrowhead_adapter = ArrowheadMassMatrix()
    mass_matrix_shape = (size,) if diagonal else (size, size)
    block_adapter.configure({("z",): mass_matrix_shape})
    arrowhead_adapter.configure({("z",): mass_matrix_shape})

    cov = torch.randn(size, size)
    cov = torch.mm(cov, cov.t())
    if diagonal:
        cov = cov.diag().diag()
    z_dist = torch.distributions.MultivariateNormal(torch.zeros(size), covariance_matrix=cov)
    g_dist = torch.distributions.MultivariateNormal(torch.zeros(size), precision_matrix=cov)
    z_samples = z_dist.sample((num_samples,)).reshape((num_samples,) + shape)
    g_samples = g_dist.sample((num_samples,)).reshape((num_samples,) + shape)

    for i in range(num_samples):
        block_adapter.update({"z": z_samples[i]}, {"z": g_samples[i]})
        arrowhead_adapter.update({"z": z_samples[i]}, {"z": g_samples[i]})
    block_adapter.end_adaptation()
    arrowhead_adapter.end_adaptation()

    assert_close(arrowhead_adapter.inverse_mass_matrix[('z',)],
                 block_adapter.inverse_mass_matrix[('z',)],
                 atol=0.3, rtol=0.3)
