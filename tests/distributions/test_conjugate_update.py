# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro.distributions as dist
from tests.common import assert_close


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
def test_beta_binomial(sample_shape, batch_shape):
    concentration1 = torch.randn(batch_shape).exp()
    concentration0 = torch.randn(batch_shape).exp()
    total = 10
    obs = dist.Binomial(total, 0.2).sample(sample_shape + batch_shape)

    f = dist.Beta(concentration1, concentration0)
    g = dist.Beta(1 + obs, 1 + total - obs)
    fg, log_normalizer = f.conjugate_update(g)

    x = fg.sample(sample_shape)
    assert_close(f.log_prob(x) + g.log_prob(x), fg.log_prob(x) + log_normalizer)


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
def test_dirichlet_multinomial(sample_shape, batch_shape):
    concentration = torch.randn(batch_shape + (3,)).exp()
    total = 10
    probs = torch.tensor([0.2, 0.3, 0.5])
    obs = dist.Multinomial(total, probs).sample(sample_shape + batch_shape)

    f = dist.Dirichlet(concentration)
    g = dist.Dirichlet(1 + obs)
    fg, log_normalizer = f.conjugate_update(g)

    x = fg.sample(sample_shape)
    assert_close(f.log_prob(x) + g.log_prob(x), fg.log_prob(x) + log_normalizer)


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
def test_gamma_poisson(sample_shape, batch_shape):
    concentration = torch.randn(batch_shape).exp()
    rate = torch.randn(batch_shape).exp()
    nobs = 5
    obs = dist.Poisson(10.).sample((nobs,) + sample_shape + batch_shape).sum(0)

    f = dist.Gamma(concentration, rate)
    g = dist.Gamma(1 + obs, nobs)
    fg, log_normalizer = f.conjugate_update(g)

    x = fg.sample(sample_shape)
    assert_close(f.log_prob(x) + g.log_prob(x), fg.log_prob(x) + log_normalizer)
