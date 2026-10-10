# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal, localcontext

import pytest
import torch

import pyro.distributions as dist

pytestmark = pytest.mark.stage("unit")


def reference_icdf(probability, asymmetry):
    # High precision avoids both power underflow and rounding to one.
    with localcontext() as context:
        context.prec = 100
        log_power = Decimal(float(probability)).ln() / Decimal(float(asymmetry))
        return float(log_power - (1 - log_power.exp()).ln())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("asymmetry", [1e-4, 0.01, 0.5, 1.0, 2.0, 1e8])
def test_skew_logistic_icdf_extreme_asymmetry(dtype, asymmetry):
    probabilities = torch.tensor([0.01, 0.5, 0.99], dtype=dtype)
    distribution = dist.SkewLogistic(
        torch.tensor(2.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(asymmetry, dtype=dtype),
    )
    expected = torch.tensor(
        [2 + 3 * reference_icdf(p, distribution.asymmetry) for p in probabilities],
        dtype=dtype,
    )
    actual = distribution.icdf(probabilities)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_skew_logistic_icdf_endpoints(dtype):
    distribution = dist.SkewLogistic(
        torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype), 0.01
    )
    actual = distribution.icdf(torch.tensor([0.0, 1.0], dtype=dtype))
    assert torch.isneginf(actual[0])
    assert torch.isposinf(actual[1])


@pytest.mark.parametrize("asymmetry", [1e-4, 1.0, 1e8])
def test_skew_logistic_icdf_gradients(asymmetry):
    probabilities = torch.tensor(
        [0.01, 0.5, 0.99], dtype=torch.float64, requires_grad=True
    )
    skew = torch.tensor(asymmetry, dtype=torch.float64, requires_grad=True)
    distribution = dist.SkewLogistic(0.0, 1.0, skew)
    actual = distribution.icdf(probabilities)
    probability_grad, skew_grad = torch.autograd.grad(
        actual.sum(), (probabilities, skew)
    )
    assert torch.isfinite(probability_grad).all()
    assert torch.isfinite(skew_grad).all()
    log_power = probabilities.detach().log() / skew.detach()
    denominator = -log_power.expm1()
    torch.testing.assert_close(
        probability_grad, 1 / (skew.detach() * probabilities.detach() * denominator)
    )
    expected_skew_grad = (
        -probabilities.detach().log() / skew.detach().square() / denominator
    ).sum()
    torch.testing.assert_close(skew_grad, expected_skew_grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_skew_logistic_icdf_broadcast(dtype):
    probabilities = torch.tensor([[0.01], [0.5], [0.99]], dtype=dtype)
    asymmetry = torch.tensor([1e-4, 1.0, 1e8], dtype=dtype)
    distribution = dist.SkewLogistic(2.0, 3.0, asymmetry)
    actual = distribution.icdf(probabilities)
    expected = torch.tensor(
        [
            [2 + 3 * reference_icdf(p, a) for a in asymmetry]
            for p in probabilities[:, 0]
        ],
        dtype=dtype,
    )
    assert actual.shape == (3, 3)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("asymmetry", [1e-4, 1e8])
@pytest.mark.init(rng_seed=123)
def test_skew_logistic_rsample_extreme_asymmetry(asymmetry):
    skew = torch.tensor(asymmetry, dtype=torch.float32, requires_grad=True)
    samples = dist.SkewLogistic(0.0, 1.0, skew).rsample((1000,))
    assert torch.isfinite(samples).all()
    gradient = torch.autograd.grad(samples.mean(), skew)[0]
    assert torch.isfinite(gradient)
