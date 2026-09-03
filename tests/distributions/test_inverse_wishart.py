# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from pyro.distributions import InverseWishart, Wishart
from tests.common import assert_close, assert_equal

# torch's Wishart.rsample emits a spurious "Singular sample detected" warning
# on valid samples (the singularity check is inverted in torch 2.13); ignore it.
pytestmark = pytest.mark.filterwarnings("ignore:Singular sample detected")


def _make_scale_matrix(p):
    # Build a symmetric positive-definite scale matrix with a known structure.
    a = torch.arange(1.0, p + 1).diag_embed()
    return a @ a.transpose(-2, -1) + torch.eye(p)


def _reference_log_prob(x, df, scale):
    # Direct implementation of the inverse Wishart log density (see, e.g.,
    # Gelman et al., Bayesian Data Analysis).
    p = scale.shape[-1]
    a = df / 2
    log_det_scale = torch.linalg.slogdet(scale).logabsdet
    log_det_x = torch.linalg.slogdet(x).logabsdet
    log_norm = (
        a * log_det_scale
        - a * p * math.log(2.0)
        - torch.special.multigammaln(a, p)
    )
    trace = (scale @ torch.linalg.inv(x)).diagonal(dim1=-2, dim2=-1).sum(-1)
    return log_norm - (df + p + 1) / 2 * log_det_x - 0.5 * trace


@pytest.mark.parametrize("p", [2, 3, 4])
def test_log_prob(p):
    torch.manual_seed(0)
    df = torch.tensor(float(p + 3))
    scale = _make_scale_matrix(p)
    d = InverseWishart(df, scale)

    for _ in range(3):
        x = d.sample()
        assert_close(d.log_prob(x), _reference_log_prob(x, df, scale), atol=1e-4)


@pytest.mark.parametrize("p", [2, 3, 4])
def test_mean_and_mode(p):
    df = torch.tensor(float(p + 5))
    scale = _make_scale_matrix(p)
    d = InverseWishart(df, scale)

    expected_mean = scale / (df - p - 1)
    expected_mode = scale / (df + p + 1)

    assert_equal(d.mean, expected_mean)
    assert_equal(d.mode, expected_mode)


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_sample_shape_and_support(p, batch_shape):
    torch.manual_seed(0)
    df = torch.tensor(float(p + 5))
    scale = _make_scale_matrix(p)
    if batch_shape:
        df = df.expand(batch_shape)
        scale = scale.expand(batch_shape + (p, p))

    d = InverseWishart(df, scale)
    x = d.sample(sample_shape=torch.Size([5]))

    expected_shape = (5,) + batch_shape + (p, p)
    assert x.shape == expected_shape
    assert (d.support.check(x) == 1).all()
    assert_close(x, x.transpose(-2, -1))  # symmetric


@pytest.mark.parametrize("p", [2, 3])
def test_inverse_is_wishart(p):
    torch.manual_seed(0)
    df = torch.tensor(float(p + 5))
    scale = _make_scale_matrix(p)

    iw = InverseWishart(df, scale)
    wishart = Wishart(df, precision_matrix=scale)

    # If X ~ InverseWishart(df, Psi) then X^{-1} ~ Wishart(df, precision=Psi),
    # so log_prob should match up to the Jacobian of the inverse map.
    x = iw.sample()
    x_inv = torch.linalg.inv(x)

    log_det_x = torch.linalg.slogdet(x).logabsdet
    assert_close(iw.log_prob(x), wishart.log_prob(x_inv) - (p + 1) * log_det_x, atol=1e-4)


def test_expand():
    df = torch.tensor(7.0)
    scale = _make_scale_matrix(2)
    d = InverseWishart(df, scale)
    expanded = d.expand(torch.Size([4]))

    assert expanded.batch_shape == torch.Size([4])
    assert expanded.event_shape == torch.Size([2, 2])
    assert expanded.scale_matrix.shape == torch.Size([4, 2, 2])
