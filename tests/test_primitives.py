# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import pyro
import pyro.distributions as dist
import torch

pytestmark = pytest.mark.stage('unit')


def test_sample_ok():
    x = pyro.sample("x", dist.Normal(0, 1))
    assert isinstance(x, torch.Tensor)
    assert x.shape == ()


def test_observe_warn():
    with pytest.warns(RuntimeWarning):
        pyro.sample("x", dist.Normal(0, 1),
                    obs=torch.tensor(0.))


def test_param_ok():
    x = pyro.param("x", torch.tensor(0.))
    assert isinstance(x, torch.Tensor)
    assert x.shape == ()


def test_deterministic_ok():
    x = pyro.deterministic("x", torch.tensor(0.))
    assert isinstance(x, torch.Tensor)
    assert x.shape == ()
