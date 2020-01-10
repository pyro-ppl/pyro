# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import scipy.special as sc
import pytest

import pyro
import pyro.distributions as dist
from pyro.infer.autoguide.utils import mean_field_entropy
from tests.common import assert_equal


def mean_field_guide(batch_tensor, design):
    # A batched variable
    w_p = pyro.param("w_p", 0.2*torch.ones(batch_tensor.shape))
    u_p = pyro.param("u_p", 0.5*torch.ones(batch_tensor.shape))
    pyro.sample("w", dist.Bernoulli(w_p))
    pyro.sample("u", dist.Bernoulli(u_p))


def h(p):
    return -(sc.xlogy(p, p) + sc.xlog1py(1 - p, -p))


@pytest.mark.parametrize("guide,args,expected_entropy", [
    (mean_field_guide, (torch.Tensor([0.]), None), torch.Tensor([h(0.2) + h(0.5)])),
    (mean_field_guide, (torch.eye(2), None), (h(0.2) + h(0.5))*torch.ones(2, 2))
])
def test_guide_entropy(guide, args, expected_entropy):
    assert_equal(mean_field_entropy(guide, args), expected_entropy)
