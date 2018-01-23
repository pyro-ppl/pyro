from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import Variable, grad

from pyro.distributions import Exponential, Gamma
from pyro.distributions.testing.rejection_exponential import RejectionExponential
from pyro.distributions.testing.rejection_gamma import RejectionStandardGamma
from tests.common import assert_equal


def compute_elbo_grad(model, guide, variables):
    x = guide.sample()
    model_log_pdf = model.batch_log_pdf(x)
    guide_log_pdf, score_function, entropy_term = guide.score_parts(x)
    log_r = model_log_pdf - guide_log_pdf
    surrogate_elbo = model_log_pdf + log_r.detach() * score_function - entropy_term
    return grad(surrogate_elbo.sum(), variables, create_graph=True)


@pytest.mark.parametrize('rate', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('factor', [0.25, 0.5, 1.0])
def test_rejector(rate, factor):
    num_samples = 100000
    rates = Variable(torch.Tensor([rate]).expand(num_samples, 1), requires_grad=True)
    factors = Variable(torch.Tensor([factor]).expand(num_samples, 1), requires_grad=True)
    dist1 = Exponential(rates)
    dist2 = RejectionExponential(rates, factors)
    x1 = dist1.sample()
    x2 = dist2.sample()
    assert_equal(x1.mean(), x2.mean(), prec=0.02, msg='bug in .sample()')
    assert_equal(x1.std(), x2.std(), prec=0.02, msg='bug in .sample()')
    assert_equal(dist1.batch_log_pdf(x1), dist2.batch_log_pdf(x1), msg='bug in .batch_log_pdf()')


@pytest.mark.xfail
@pytest.mark.parametrize('rate', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('factor', [0.25, 0.5, 1.0])
def test_exponential_elbo(rate, factor):
    num_samples = 100000
    rates = Variable(torch.Tensor([rate]).expand(num_samples, 1), requires_grad=True)
    factors = Variable(torch.Tensor([factor]).expand(num_samples, 1), requires_grad=True)

    model = Exponential(Variable(torch.ones(num_samples, 1)))
    guide1 = Exponential(rates)
    guide2 = RejectionExponential(rates, factors)

    grads = []
    for guide in [guide1, guide2]:
        grads.append(compute_elbo_grad(model, guide, [rates])[0].data)
    expected, actual = grads
    assert_equal(actual.mean(), expected.mean(), prec=0.05, msg='bad grad for rate')

    # FIXME this fails
    actual = compute_elbo_grad(model, guide2, [factors])[0].data
    assert_equal(actual.mean(), 0.0, prec=0.05, msg='bad grad for factor')


@pytest.mark.parametrize('alpha', [1.0, 2.0, 5.0])
def test_standard_gamma_elbo(alpha):
    num_samples = 100000
    alphas = Variable(torch.Tensor([alpha]).expand(num_samples, 1), requires_grad=True)
    betas = Variable(torch.ones(num_samples, 1))

    model = Gamma(Variable(torch.ones(num_samples, 1)), betas)
    guide1 = Gamma(alphas, betas)
    guide2 = RejectionStandardGamma(alphas)  # implemented using Rejector

    grads = []
    for guide in [guide1, guide2]:
        grads.append(compute_elbo_grad(model, guide, [alphas])[0].data)
    expected, actual = grads
    assert_equal(actual.mean(), expected.mean(), prec=0.01, msg='bad grad for alpha')
