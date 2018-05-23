from __future__ import absolute_import, division, print_function

import numpy as np
import torch

import pytest
from pyro.distributions import AVFMultivariateNormal, MultivariateNormal, OMTMultivariateNormal
from tests.common import assert_equal


def analytic_grad(L11=1.0, L22=1.0, L21=1.0, omega1=1.0, omega2=1.0):
    dp = L11 * omega1 + L21 * omega2
    fact_1 = - omega2 * dp
    fact_2 = np.exp(- 0.5 * (L22 * omega2) ** 2)
    fact_3 = np.exp(- 0.5 * dp ** 2)
    return fact_1 * fact_2 * fact_3


@pytest.mark.parametrize('L21', [0.4, 1.1])
@pytest.mark.parametrize('L11', [0.6])
@pytest.mark.parametrize('omega1', [0.5])
@pytest.mark.parametrize('sample_shape', [torch.Size([1000, 2000]), torch.Size([200000])])
@pytest.mark.parametrize('k', [1])
@pytest.mark.parametrize('mvn_dist', ['OMTMultivariateNormal', 'AVFMultivariateNormal'])
def test_mean_gradient(mvn_dist, k, sample_shape, L21, omega1, L11, L22=0.8, L33=0.9, omega2=0.75):
    if mvn_dist == 'OMTMultivariateNormal' and k > 1:
        return

    omega = torch.tensor([omega1, omega2, 0.0])
    loc = torch.zeros(3, requires_grad=True)
    zero_vec = [0.0, 0.0, 0.0]
    off_diag = torch.tensor([zero_vec, [L21, 0.0, 0.0], zero_vec], requires_grad=True)
    L = torch.diag(torch.tensor([L11, L22, L33])) + off_diag

    if mvn_dist == 'OMTMultivariateNormal':
        dist = OMTMultivariateNormal(loc, L)
    elif mvn_dist == 'AVFMultivariateNormal':
        CV = torch.tensor(1.1 * torch.rand(2, k, 3), requires_grad=True)
        dist = AVFMultivariateNormal(loc, L, CV)

    z = dist.rsample(sample_shape)
    torch.cos((omega*z).sum(-1)).mean().backward()

    computed_grad = off_diag.grad.cpu().data.numpy()[1, 0]
    analytic = analytic_grad(L11=L11, L22=L22, L21=L21, omega1=omega1, omega2=omega2)
    assert(off_diag.grad.size() == off_diag.size())
    assert(loc.grad.size() == loc.size())
    assert(torch.triu(off_diag.grad, 1).sum() == 0.0)
    assert_equal(analytic, computed_grad, prec=0.005,
                 msg='bad cholesky grad for %s (expected %.5f, got %.5f)' %
                 (mvn_dist, analytic, computed_grad))


@pytest.mark.skip(reason="Slow; tests to be run when refactoring")
@pytest.mark.parametrize('L21', [0.4, 1.1])
@pytest.mark.parametrize('L11', [0.6, 0.95])
@pytest.mark.parametrize('omega1', [0.5, 0.9])
@pytest.mark.parametrize('k', [3])
@pytest.mark.parametrize('mvn_dist', ['OMTMultivariateNormal', 'AVFMultivariateNormal'])
def test_mean_single_gradient(mvn_dist, k, L21, omega1, L11, L22=0.8, L33=0.9, omega2=0.75, n_samples=20000):
    omega = torch.tensor([omega1, omega2, 0.0])
    loc = torch.zeros(3, requires_grad=True)
    zero_vec = [0.0, 0.0, 0.0]
    off_diag = torch.tensor([zero_vec, [L21, 0.0, 0.0], zero_vec], requires_grad=True)
    L = torch.diag(torch.tensor([L11, L22, L33])) + off_diag

    if mvn_dist == 'OMTMultivariateNormal':
        dist = OMTMultivariateNormal(loc, L)
    elif mvn_dist == 'AVFMultivariateNormal':
        CV = torch.tensor(0.2 * torch.rand(2, k, 3), requires_grad=True)
        dist = AVFMultivariateNormal(loc, L, CV)

    computed_grads = []

    for _ in range(n_samples):
        z = dist.rsample()
        torch.cos((omega*z).sum(-1)).mean().backward()
        assert(off_diag.grad.size() == off_diag.size())
        assert(loc.grad.size() == loc.size())
        assert(torch.triu(off_diag.grad, 1).sum() == 0.0)

        computed_grad = off_diag.grad.cpu()[1, 0].item()
        computed_grads.append(computed_grad)
        off_diag.grad.zero_()
        loc.grad.zero_()

    computed_grad = np.mean(computed_grads)
    analytic = analytic_grad(L11=L11, L22=L22, L21=L21, omega1=omega1, omega2=omega2)
    assert_equal(analytic, computed_grad, prec=0.01,
                 msg='bad cholesky grad for %s (expected %.5f, got %.5f)' % (mvn_dist, analytic, computed_grad))


@pytest.mark.parametrize('mvn_dist', [OMTMultivariateNormal, AVFMultivariateNormal])
def test_log_prob(mvn_dist):
    loc = torch.tensor([2.0, 1.0, 1.0, 2.0, 2.0])
    D = torch.tensor([1.0, 2.0, 3.0, 1.0, 3.0])
    W = torch.tensor([[1.0, -1.0, 2.0, 2.0, 4.0], [2.0, 1.0, 1.0, 2.0, 6.0]])
    x = torch.tensor([2.0, 3.0, 4.0, 1.0, 7.0])
    L = D.diag() + torch.tril(W.t().matmul(W))
    cov = torch.mm(L, L.t())

    mvn = MultivariateNormal(loc, cov)
    if mvn_dist == OMTMultivariateNormal:
        mvn_prime = OMTMultivariateNormal(loc, L)
    elif mvn_dist == AVFMultivariateNormal:
        CV = torch.tensor(0.2 * torch.rand(2, 2, 5))
        mvn_prime = AVFMultivariateNormal(loc, L, CV)
    assert_equal(mvn.log_prob(x), mvn_prime.log_prob(x))
