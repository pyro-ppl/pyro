# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_close


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
def test_multivariate_log_prob(batch_shape, sample_shape):
    size = 10
    logits = torch.randn(batch_shape + (size,)).mul(0.1).exp()
    d1 = dist.Multinomial(logits=logits, validate_args=False)
    d2 = torch.distributions.Multinomial(logits=logits, validate_args=False)

    for total_count in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        value = dist.Multinomial(total_count, logits=logits).sample(sample_shape)
        actual = d1.log_prob(value)
        expected = d2.log_prob(value)
        assert_close(actual, expected)
