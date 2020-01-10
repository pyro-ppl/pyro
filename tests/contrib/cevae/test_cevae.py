# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import io
import warnings

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.cevae import CEVAE, DistributionNet
from tests.common import assert_close

DIST_NETS = [cls.__name__.lower()[:-3]
             for cls in DistributionNet.__subclasses__()]


def generate_data(num_data, feature_dim):
    z = dist.Bernoulli(0.5).sample([num_data])
    x = dist.Normal(z, 5 * z + 3 * (1 - z)).sample([feature_dim]).t()
    t = dist.Bernoulli(0.75 * z + 0.25 * (1 - z)).sample()
    y = dist.Bernoulli(logits=3 * (z + 2 * (2 * t - 2))).sample()
    return x, t, y


@pytest.mark.parametrize("num_data", [1, 100, 200])
@pytest.mark.parametrize("feature_dim", [1, 2])
@pytest.mark.parametrize("outcome_dist", DIST_NETS)
def test_smoke(num_data, feature_dim, outcome_dist):
    x, t, y = generate_data(num_data, feature_dim)
    if outcome_dist == "exponential":
        y.clamp_(min=1e-20)
    cevae = CEVAE(feature_dim, outcome_dist)
    cevae.fit(x, t, y, num_epochs=2)
    ite = cevae.ite(x)
    assert ite.shape == (num_data,)


@pytest.mark.parametrize("feature_dim", [1, 2])
@pytest.mark.parametrize("outcome_dist", DIST_NETS)
@pytest.mark.parametrize("jit", [False, True], ids=["python", "jit"])
def test_serialization(jit, feature_dim, outcome_dist):
    x, t, y = generate_data(num_data=32, feature_dim=feature_dim)
    if outcome_dist == "exponential":
        y.clamp_(min=1e-20)
    cevae = CEVAE(feature_dim, outcome_dist=outcome_dist, num_samples=1000, hidden_dim=32)
    cevae.fit(x, t, y, num_epochs=4, batch_size=8)
    pyro.set_rng_seed(0)
    expected_ite = cevae.ite(x)

    if jit:
        traced_cevae = cevae.to_script_module()
        f = io.BytesIO()
        torch.jit.save(traced_cevae, f)
        f.seek(0)
        loaded_cevae = torch.jit.load(f)
    else:
        f = io.BytesIO()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            torch.save(cevae, f)
        f.seek(0)
        loaded_cevae = torch.load(f)

    pyro.set_rng_seed(0)
    actual_ite = loaded_cevae.ite(x)
    assert_close(actual_ite, expected_ite, atol=0.1)
