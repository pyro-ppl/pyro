from __future__ import absolute_import, division, print_function

import pytest

import numpy as np

import torch
from torch.autograd import Variable

from pyro.distributions.omt_mvn import OMTMultivariateNormal, OTCVMultivariateNormal
from pyro.distributions import MultivariateNormal

from tests.common import assert_equal


def analytic_grad(L11=1.0, L22=1.0, L21=1.0, omega1=1.0, omega2=1.0):
    dp = L11 * omega1 + L21 * omega2
    fact_1 = - omega2 * dp
    fact_2 = np.exp(- 0.5 * (L22 * omega2) ** 2)
    fact_3 = np.exp(- 0.5 * dp ** 2)
    return fact_1 * fact_2 * fact_3


@pytest.mark.parametrize('L21', [0.4, 1.1])
@pytest.mark.parametrize('L11', [0.6, 0.95])
@pytest.mark.parametrize('omega1', [0.5, 0.9])
@pytest.mark.parametrize('sample_shape', [torch.Size([1000, 1000]), torch.Size([100000])])
@pytest.mark.parametrize('mvn_dist', [OMTMultivariateNormal])
#@pytest.mark.parametrize('mvn_dist', [OTCVMultivariateNormal])
#@pytest.mark.parametrize('mvn_dist', [OMTMultivariateNormal, OTCVMultivariateNormal])
def test_mean(mvn_dist, sample_shape, L21, omega1, L11, L22=0.8, L33=0.9, omega2=0.75):
    dim = 3
    omega = Variable(torch.Tensor([omega1, omega2, 0.0]))
    mu = Variable(torch.zeros(dim), requires_grad=True)
    #off_diag = Variable(L21 * torch.ones(dim, dim), requires_grad=True)
    zero_vec = [0.0, 0.0, 0.0]
    off_diag = Variable(torch.Tensor([zero_vec, [L21, 0.0, 0.0], zero_vec]), requires_grad=True)
    #print("off_diag", off_diag)
    #L = Variable(torch.diag(torch.Tensor([L11, L22]))) + torch.tril(off_diag, -1)
    L = Variable(torch.diag(torch.Tensor([L11, L22, L33]))) + off_diag
    #print("L", L, torch.mm(L, L.t()))

    if mvn_dist==OMTMultivariateNormal:
        dist = mvn_dist(mu, L)
    else:
        noise = 0.3
        B = Variable(noise * torch.randn(dim, dim))
        C = Variable(noise * torch.randn(dim, dim))
        D = noise * torch.randn(dim, dim)
        D = Variable(D - D.t())
        F = Variable(noise * torch.randn(dim, dim))
        dist = mvn_dist(mu, L, B, C, D, F)
    z = dist.rsample(sample_shape)
    torch.cos((omega*z).sum(-1)).mean().backward()

    computed_grad = off_diag.grad.data.numpy()[1, 0]
    analytic = analytic_grad(L11=L11, L22=L22, L21=L21, omega1=omega1, omega2=omega2)
    return computed_grad
    #print("an comp", analytic, computed_grad, L21, omega1, L11, sample_shape)
    assert(mu.grad.size() == mu.size())
    assert(off_diag.grad.size() == off_diag.size())
    #assert_equal(analytic, computed_grad, prec=0.015, msg='bad cholesky grad for %s (expected %.5f, got %.5f)' % (mvn_dist,
    #             analytic, computed_grad))

blah=[]
for _ in range(20000):
    blah.append(test_mean(OMTMultivariateNormal, torch.Size(), L21=0.9, omega1=1.3, L11=1.2))
print("computed", np.mean(blah), np.std(blah))
print("analytic", analytic_grad(L11=1.2, L22=0.8, L21=0.9, omega1=1.3, omega2=0.75))

#test_mean(OTCVMultivariateNormal, torch.Size([2000, 2000]), 0.5, 1.0, 0.7)
#test_mean(OMTMultivariateNormal, torch.Size([20000]), 0.9, 1.3, 1.2)
#test_mean(OMTMultivariateNormal, torch.Size([2000, 10]), 0.5, 1.0, 0.7)

def test_log_prob():
    loc = Variable(torch.Tensor([2, 1, 1, 2, 2]))
    D = Variable(torch.Tensor([1, 2, 3, 1, 3]))
    W = Variable(torch.Tensor([[1, -1, 2, 2, 4], [2, 1, 1, 2, 6]]))
    x = Variable(torch.Tensor([2, 3, 4, 1, 7]))
    L = D.diag() + torch.tril(W.t().matmul(W))
    cov = torch.mm(L, L.t())

    mvn = MultivariateNormal(loc, cov)
    omt_mvn = OMTMultivariateNormal(loc, L)
    assert_equal(mvn.log_prob(x), omt_mvn.log_prob(x))
