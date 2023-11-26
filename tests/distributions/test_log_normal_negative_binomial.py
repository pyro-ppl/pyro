# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.distributions import LogNormalNegativeBinomial
from tests.common import assert_close


@pytest.mark.parametrize("num_quad_points", [2, 4])
@pytest.mark.parametrize("shape", [(2,), (4, 3)])
def test_lnnb_shapes(num_quad_points, shape):
    logits = torch.randn(shape)
    total_count = 5.0
    multiplicative_noise_scale = torch.rand(shape)

    d = LogNormalNegativeBinomial(
        total_count, logits, multiplicative_noise_scale, num_quad_points=num_quad_points
    )

    assert d.batch_shape == shape
    assert d.log_prob(torch.ones(shape)).shape == shape

    assert d.expand(shape + shape).batch_shape == shape + shape
    assert d.expand(shape + shape).log_prob(torch.ones(shape)).shape == shape + shape


@pytest.mark.parametrize("total_count", [0.5, 4.0])
@pytest.mark.parametrize("multiplicative_noise_scale", [0.01, 0.25])
def test_lnnb_mean_variance(
    total_count, multiplicative_noise_scale, num_quad_points=128, N=512
):
    logits = torch.tensor(2.0)
    d = LogNormalNegativeBinomial(
        total_count, logits, multiplicative_noise_scale, num_quad_points=num_quad_points
    )

    values = torch.arange(N)
    probs = d.log_prob(values).exp()
    assert_close(1.0, probs.sum().item(), atol=1.0e-6)

    expected_mean = (probs * values).sum()
    assert_close(expected_mean, d.mean, atol=1.0e-6, rtol=1.0e-5)

    expected_var = (probs * (values - d.mean).pow(2.0)).sum()
    assert_close(expected_var, d.variance, atol=1.0e-6, rtol=1.0e-5)
