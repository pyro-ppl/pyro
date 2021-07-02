# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.distributions import constraints
from pyro.ops.tensor_utils import safe_normalize


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000])
def test_sphere_check(dim):
    data = torch.randn(100, dim)
    assert not constraints.sphere.check(data).any()

    data = safe_normalize(data)
    actual = constraints.sphere.check(data)
    assert actual.all()
    assert actual.shape == data.shape[:-1]


@pytest.mark.parametrize("batch_shape", [(), (3, 4)])
@pytest.mark.parametrize(
    "constraint, event_shape",
    [
        (constraints.positive_ordered_vector, (5,)),
        (constraints.corr_matrix, (6,)),
        (constraints.positive_definite, (3, 3)),
    ],
)
def test_constraints(constraint, batch_shape, event_shape):
    x = torch.randn(batch_shape + event_shape)
    y = torch.distributions.transform_to(constraint)(x)
    actual = constraint.check(y)
    assert actual.all()
    assert actual.shape == batch_shape
