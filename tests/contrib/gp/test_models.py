from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.models import GPRegression, SparseGPRegression
from pyro.contrib.gp import InducingPoints
from tests.common import assert_equal


def test_forward_gpr():
    kernel = RBF(input_dim=3, variance=torch.ones(1), lengthscale=torch.ones(3))
    X = Variable(torch.Tensor([[1, 2, 3], [4, 5, 6]]))
    y = Variable(torch.Tensor([0, 1]))
    # hacky: noise ~ 0, Xnew = X
    Xnew = X
    noise = torch.Tensor([1e-6])

    gpr = GPRegression(X, y, kernel, noise=noise)
    loc, cov = gpr(Xnew, full_cov=True)

    assert loc.dim() == 1
    assert cov.dim() == 2
    assert loc.size(0) == 2
    assert cov.size(0) == 2
    assert cov.size(1) == 2
    assert_equal(loc, y)
    assert_equal(cov.data.abs().sum(), 0)


def test_forward_sgpr():
    kernel = RBF(input_dim=3, variance=torch.ones(2), lengthscale=torch.ones(2))
    X = Variable(torch.Tensor([[1, 5, 3], [4, 3, 7]]))
    y = Variable(torch.Tensor([0, 1]))
    # hacky: noise ~ 0, Xnew = Xu = X
    Xu = InducingPoints(X.data.clone())
    Xnew = X
    noise = torch.Tensor([1e-6])

    sgpr = SparseGPRegression(X, y, kernel, Xu, noise=noise)
    loc, cov = sgpr(Xnew, full_cov=True)

    assert loc.dim() == 1
    assert cov.dim() == 2
    assert loc.size(0) == 2
    assert cov.size(0) == 2
    assert cov.size(1) == 2
    assert_equal(loc, y)
    assert_equal(cov.data.abs().sum(), 0)


def test_forward_sgpr_vs_gpr():
    kernel = RBF(input_dim=3, variance=torch.ones(2), lengthscale=torch.ones(2))
    X = Variable(torch.Tensor([[2, 5, 3], [4, 3, 7]]))
    y = Variable(torch.Tensor([0, 1]))
    Xu = InducingPoints(X.data.clone())  # must be set to compare
    Xnew = Variable(torch.Tensor([[3, 1, 4], [1, 3, 1]]))
    noise = torch.Tensor([1])

    gpr = GPRegression(X, y, kernel, noise=noise)
    sgpr = SparseGPRegression(X, y, kernel, Xu, noise=noise)
    sgpr_fitc = SparseGPRegression(X, y, kernel, Xu, approx="FITC", noise=noise)

    loc_gpr, cov_gpr = gpr(Xnew, full_cov=True)
    loc_sgpr, cov_sgpr = sgpr(Xnew, full_cov=True)
    loc_fitc, cov_fitc = sgpr_fitc(Xnew, full_cov=True)

    loc_gpr, sd_gpr = gpr(Xnew, full_cov=False)
    loc_sgpr, sd_sgpr = sgpr(Xnew, full_cov=False)

    assert_equal(loc_gpr, loc_sgpr)
    assert_equal(cov_gpr, cov_sgpr, prec=1e-4)
    assert_equal(cov_sgpr, cov_fitc, prec=1e-4)
    assert_equal(sd_gpr, sd_sgpr, prec=1e-4)
