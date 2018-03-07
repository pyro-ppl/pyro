from __future__ import absolute_import, division, print_function

import torch

from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.models import GPRegression, SparseGPRegression
from tests.common import assert_equal


def test_forward_gpr():
    kernel = RBF(input_dim=3, variance=torch.tensor([1]), lengthscale=torch.tensor([3]))
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([0, 1])
    # hacky: noise ~ 0, Xnew = X
    Xnew = X
    noise = torch.tensor([1e-6])

    gpr = GPRegression(X, y, kernel, noise=noise)

    loc, cov = gpr(Xnew, full_cov=True)

    assert loc.dim() == 1
    assert cov.dim() == 2
    assert loc.size(0) == 2
    assert cov.size(0) == 2
    assert cov.size(1) == 2
    assert_equal(loc, y)
    assert_equal(cov.abs().sum().item(), 0)


def test_forward_sgpr():
    kernel = RBF(input_dim=3, variance=torch.tensor([2]), lengthscale=torch.tensor([2]))
    X = torch.tensor([[1, 5, 3], [4, 3, 7]])
    y = torch.tensor([0, 1])
    # hacky: noise ~ 0, Xnew = Xu = X
    Xu = X.data
    Xnew = X
    noise = torch.tensor([1e-6])

    sgpr = SparseGPRegression(X, y, kernel, Xu, noise=noise)

    loc, cov = sgpr(Xnew, full_cov=True)

    assert loc.dim() == 1
    assert cov.dim() == 2
    assert loc.size(0) == 2
    assert cov.size(0) == 2
    assert cov.size(1) == 2
    assert_equal(loc, y)
    assert_equal(cov.abs().sum().item(), 0)


def test_forward_sgpr_vs_gpr():
    kernel = RBF(input_dim=3, variance=torch.tensor([2]), lengthscale=torch.tensor([2]))
    X = torch.tensor([[2, 5, 3], [4, 3, 7]])
    y = torch.tensor([0, 1])
    Xu = X.data  # must be set to compare
    Xnew = torch.tensor([[3, 1, 4], [1, 3, 1]])
    noise = torch.tensor([1])

    gpr = GPRegression(X, y, kernel, noise=noise)
    sgpr = SparseGPRegression(X, y, kernel, Xu, noise=noise)
    sgpr_fitc = SparseGPRegression(X, y, kernel, Xu, approx="FITC", noise=noise)

    loc_gpr, cov_gpr = gpr(Xnew, full_cov=True)
    loc_sgpr, cov_sgpr = sgpr(Xnew, full_cov=True)
    loc_fitc, cov_fitc = sgpr_fitc(Xnew, full_cov=True)

    loc_gpr, sd_gpr = gpr(Xnew, full_cov=False)
    loc_sgpr, sd_sgpr = sgpr(Xnew, full_cov=False)

    assert_equal(loc_gpr, loc_sgpr)
    assert_equal(cov_gpr, cov_sgpr)
    assert_equal(cov_sgpr, cov_fitc)
    assert_equal(sd_gpr, sd_sgpr)
