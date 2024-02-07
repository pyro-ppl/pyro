# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro import poutine

pytestmark = pytest.mark.stage("unit")


def test_sample_ok():
    x = pyro.sample("x", dist.Normal(0, 1))
    assert isinstance(x, torch.Tensor)
    assert x.shape == ()


def test_observe_warn():
    with pytest.warns(RuntimeWarning):
        pyro.sample("x", dist.Normal(0, 1), obs=torch.tensor(0.0))


def test_param_ok():
    x = pyro.param("x", torch.tensor(0.0))
    assert isinstance(x, torch.Tensor)
    assert x.shape == ()


def test_deterministic_ok():
    x = pyro.deterministic("x", torch.tensor(0.0))
    assert isinstance(x, torch.Tensor)
    assert x.shape == ()


@pytest.mark.parametrize(
    "mask",
    [
        None,
        torch.tensor(True),
        torch.tensor([True]),
        torch.tensor([True, False, True]),
    ],
)
def test_obs_mask_shape(mask: Optional[torch.Tensor]):
    data = torch.randn(3, 2)

    def model():
        with pyro.plate("data", 3):
            pyro.sample(
                "y",
                dist.MultivariateNormal(torch.zeros(2), scale_tril=torch.eye(2)),
                obs=data,
                obs_mask=mask,
            )

    trace = poutine.trace(model).get_trace()
    y_dist = trace.nodes["y"]["fn"]
    assert y_dist.batch_shape == (3,)
    assert y_dist.event_shape == (2,)
