# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.infer.mcmc.adaptation import (
    _arrowhead_sqrt,
    _arrowhead_sqrt_inverse_to_inverse,
    _arrowhead_sqrt_matmul,
    _arrowhead_sqrt_to_sqrt_inverse,
    ArrowheadMatrix,
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
    arrowhead_adapter = ArrowheadMassMatrix(head_size=0 if diagonal else size)
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


@pytest.mark.parametrize('head_size', [0, 2, 5])
def test_arrowhead_utilities(head_size):
    size = 5
    cov = torch.randn(size, size)
    cov = torch.mm(cov, cov.t())

    mask = torch.ones(size, size)
    mask[head_size:, head_size:] = 0.
    mask.view(-1)[::size + 1][head_size:] = 1.
    arrowhead_full = mask * cov
    expected = torch.flip(torch.flip(arrowhead_full, (-2, -1)).cholesky(), (-2, -1))
    # test if those flip ops give expected upper triangular values
    assert_close(expected.triu(), expected)
    assert_close(expected.matmul(expected.t()), arrowhead_full)

    # test arrowhead_sqrt
    arrowhead = ArrowheadMatrix(cov[:head_size], cov.diag()[head_size:])
    actual = _arrowhead_sqrt(arrowhead)
    assert_close(actual.top, expected[:head_size])
    assert_close(actual.bottom_diag, expected.diag()[head_size:])

    # test arrowhead_sqrt_inverse
    expected = expected.inverse()
    actual = _arrowhead_sqrt_to_sqrt_inverse(actual)
    assert_close(actual.top, expected[:head_size])
    assert_close(actual.bottom_diag, expected.diag()[head_size:])

    # test arrowhead_sqrt_matmul
    v = torch.randn(size)
    assert_close(_arrowhead_sqrt_matmul(actual, v), expected.matmul(v))
    assert_close(_arrowhead_sqrt_matmul(actual, v, transpose=True),
                 expected.t().matmul(v))

    # test arrowhead_sqrt_inverse_to_inverse
    actual = _arrowhead_sqrt_inverse_to_inverse(actual)
    expected = arrowhead_full.inverse() if head_size > 0 else arrowhead_full.diag().reciprocal()
    assert_close(actual, expected)
