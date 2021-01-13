# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.distributions import constraints
from pyro.distributions.projected_normal import safe_project


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000])
def test_shpere_check(dim):
    data = torch.randn(100, dim)
    assert not constraints.sphere.check(data).any()

    data = safe_project(data)
    actual = constraints.sphere.check(data)
    assert actual.all()
    assert actual.shape == data.shape[:-1]
