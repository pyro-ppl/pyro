# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from tests.common import default_dtype


@pytest.mark.parametrize("strength", [0, 1, 10, 100, 1000])
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
def test_log_prob(dtype, dim, strength):
    with default_dtype(dtype):
        concentration = torch.full((dim,), float(strength), requires_grad=True)
        value = dist.ProjectedNormal(torch.zeros_like(concentration)).sample([10000])
        d = dist.ProjectedNormal(concentration)

        logp = d.log_prob(value)
        assert logp.max().lt(1 + dim * strength).all()

        logp.sum().backward()
        assert not torch.isnan(concentration.grad).any()
