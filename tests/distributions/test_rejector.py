# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

from pyro.distributions import Exponential, Gamma
from pyro.distributions.testing.rejection_exponential import RejectionExponential
from pyro.distributions.testing.rejection_gamma import (RejectionGamma, RejectionStandardGamma, ShapeAugmentedBeta,
                                                        ShapeAugmentedGamma)
from tests.common import assert_equal

SIZES = list(map(torch.Size, [[], [1], [2], [3], [1, 1], [1, 2], [2, 3, 4]]))


@pytest.mark.parametrize('sample_shape', SIZES)
@pytest.mark.parametrize('batch_shape', filter(bool, SIZES))
def test_rejection_standard_gamma_sample_shape(sample_shape, batch_shape):
    alphas = torch.ones(batch_shape)
    dist = RejectionStandardGamma(alphas)
    x = dist.rsample(sample_shape)
    assert x.shape == sample_shape + batch_shape


@pytest.mark.parametrize('sample_shape', SIZES)
@pytest.mark.parametrize('batch_shape', filter(bool, SIZES))
def test_rejection_exponential_sample_shape(sample_shape, batch_shape):
    rates = torch.ones(batch_shape)
    factors = torch.ones(batch_shape) * 0.5
    dist = RejectionExponential(rates, factors)
    x = dist.rsample(sample_shape)
    assert x.shape == sample_shape + batch_shape


def compute_elbo_grad(model, guide, variables):
    x = guide.rsample()
    model_log_prob = model.log_prob(x)
    guide_log_prob, score_function, entropy_term = guide.score_parts(x)
    log_r = model_log_prob - guide_log_prob
    surrogate_elbo = model_log_prob + log_r.detach() * score_function - entropy_term
    return grad(surrogate_elbo.sum(), variables, create_graph=True)


@pytest.mark.parametrize('rate', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('factor', [0.25, 0.5, 1.0])
def test_rejector(rate, factor):
    num_samples = 100000
    rates = torch.tensor(rate).expand(num_samples, 1)
    factors = torch.tensor(factor).expand(num_samples, 1)

    dist1 = Exponential(rates)
    dist2 = RejectionExponential(rates, factors)  # implemented using Rejector
    x1 = dist1.rsample()
    x2 = dist2.rsample()
    assert_equal(x1.mean(), x2.mean(), prec=0.02, msg='bug in .rsample()')
    assert_equal(x1.std(), x2.std(), prec=0.02, msg='bug in .rsample()')
    assert_equal(dist1.log_prob(x1), dist2.log_prob(x1), msg='bug in .log_prob()')


@pytest.mark.parametrize('rate', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('factor', [0.25, 0.5, 1.0])
def test_exponential_elbo(rate, factor):
    num_samples = 100000
    rates = torch.full((num_samples, 1), rate).requires_grad_()
    factors = torch.full((num_samples, 1), factor).requires_grad_()
    model = Exponential(torch.ones(num_samples, 1))
    guide1 = Exponential(rates)
    guide2 = RejectionExponential(rates, factors)  # implemented using Rejector

    grads = []
    for guide in [guide1, guide2]:
        grads.append(compute_elbo_grad(model, guide, [rates])[0])
    expected, actual = grads
    assert_equal(actual.mean(), expected.mean(), prec=0.05, msg='bad grad for rate')

    actual = compute_elbo_grad(model, guide2, [factors])[0]
    assert_equal(actual.mean().item(), 0.0, prec=0.05, msg='bad grad for factor')


@pytest.mark.parametrize('alpha', [1.0, 2.0, 5.0])
def test_standard_gamma_elbo(alpha):
    num_samples = 100000
    alphas = torch.full((num_samples, 1), alpha).requires_grad_()
    betas = torch.ones(num_samples, 1)

    model = Gamma(torch.ones(num_samples, 1), betas)
    guide1 = Gamma(alphas, betas)
    guide2 = RejectionStandardGamma(alphas)  # implemented using Rejector

    grads = []
    for guide in [guide1, guide2]:
        grads.append(compute_elbo_grad(model, guide, [alphas])[0].data)
    expected, actual = grads
    assert_equal(actual.mean(), expected.mean(), prec=0.01, msg='bad grad for alpha')


@pytest.mark.parametrize('alpha', [1.0, 2.0, 5.0])
@pytest.mark.parametrize('beta', [0.2, 0.5, 1.0, 2.0, 5.0])
def test_gamma_elbo(alpha, beta):
    num_samples = 100000
    alphas = torch.full((num_samples, 1), alpha).requires_grad_()
    betas = torch.full((num_samples, 1), beta).requires_grad_()

    model = Gamma(torch.ones(num_samples, 1), torch.ones(num_samples, 1))
    guide1 = Gamma(alphas, betas)
    guide2 = RejectionGamma(alphas, betas)  # implemented using Rejector

    grads = []
    for guide in [guide1, guide2]:
        grads.append(compute_elbo_grad(model, guide, [alphas, betas]))
    expected, actual = grads
    expected = [g.mean() for g in expected]
    actual = [g.mean() for g in actual]
    scale = [(1 + abs(g)) for g in expected]
    assert_equal(actual[0] / scale[0], expected[0] / scale[0], prec=0.01, msg='bad grad for alpha')
    assert_equal(actual[1] / scale[1], expected[1] / scale[1], prec=0.01, msg='bad grad for beta')


@pytest.mark.parametrize('alpha', [0.2, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize('beta', [0.2, 0.5, 1.0, 2.0, 5.0])
def test_shape_augmented_gamma_elbo(alpha, beta):
    num_samples = 100000
    alphas = torch.full((num_samples, 1), alpha).requires_grad_()
    betas = torch.full((num_samples, 1), beta).requires_grad_()

    model = Gamma(torch.ones(num_samples, 1), torch.ones(num_samples, 1))
    guide1 = Gamma(alphas, betas)
    guide2 = ShapeAugmentedGamma(alphas, betas)  # implemented using Rejector

    grads = []
    for guide in [guide1, guide2]:
        grads.append(compute_elbo_grad(model, guide, [alphas, betas]))
    expected, actual = grads
    expected = [g.mean() for g in expected]
    actual = [g.mean() for g in actual]
    scale = [(1 + abs(g)) for g in expected]
    assert_equal(actual[0] / scale[0], expected[0] / scale[0], prec=0.05, msg='bad grad for alpha')
    assert_equal(actual[1] / scale[1], expected[1] / scale[1], prec=0.05, msg='bad grad for beta')


@pytest.mark.parametrize('alpha', [0.5, 1.0, 4.0])
@pytest.mark.parametrize('beta', [0.5, 1.0, 4.0])
def test_shape_augmented_beta(alpha, beta):
    num_samples = 10000
    alphas = torch.full((num_samples, 1), alpha).requires_grad_()
    betas = torch.full((num_samples, 1), beta).requires_grad_()
    dist = ShapeAugmentedBeta(alphas, betas)  # implemented using Rejector
    z = dist.rsample()
    cost = z.sum()
    (cost + cost.detach() * dist.score_parts(z)[1]).backward()
    mean_alpha_grad = alphas.grad.mean().item()
    mean_beta_grad = betas.grad.mean().item()
    expected_alpha_grad = beta / (alpha + beta) ** 2
    expected_beta_grad = -alpha / (alpha + beta) ** 2
    assert_equal(mean_alpha_grad, expected_alpha_grad, prec=0.02, msg='bad grad for alpha')
    assert_equal(mean_beta_grad, expected_beta_grad, prec=0.02, msg='bad grad for beta')
