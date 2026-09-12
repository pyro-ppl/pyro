# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import pyro
import pyro.distributions as dist
from tests.common import assert_equal


@pytest.mark.parametrize(
    "batch_shape,expanded_shape", [((), (4,)), ((1,), (4,)), ((2, 1), (2, 4))]
)
@pytest.mark.parametrize("validate_args", [False, True])
def test_soft_asymmetric_laplace_expand(batch_shape, expanded_shape, validate_args):
    loc = torch.arange(math.prod(batch_shape), dtype=torch.get_default_dtype()).reshape(
        batch_shape
    )
    original = dist.SoftAsymmetricLaplace(
        loc, 1.5, 1.7, 0.4, validate_args=validate_args
    )
    expanded = original.expand(expanded_shape)
    expected = dist.SoftAsymmetricLaplace(loc.expand(expanded_shape), 1.5, 1.7, 0.4)

    assert isinstance(expanded, dist.SoftAsymmetricLaplace)
    assert original.batch_shape == batch_shape
    assert expanded.batch_shape == expanded_shape
    assert expanded._validate_args == validate_args
    assert_equal(expanded.mean, expected.mean)
    assert_equal(expanded.variance, expected.variance)
    samples = expanded.rsample((5,))
    assert samples.shape == (5,) + expanded_shape
    assert_equal(expanded.log_prob(samples), expected.log_prob(samples))
    repeated = expanded.expand((3,) + expanded_shape)
    assert_equal(
        repeated.log_prob(samples[0]),
        expanded.log_prob(samples[0]).expand((3,) + expanded_shape),
    )


def test_soft_asymmetric_laplace_plate():
    loc = torch.tensor(0.5, requires_grad=True)
    with pyro.plate("data", 4):
        sample = pyro.sample("obs", dist.SoftAsymmetricLaplace(loc, 1.5, 1.7, 0.4))
    assert sample.shape == (4,)
    sample.sum().backward()
    assert_equal(loc.grad, torch.tensor(4.0))
