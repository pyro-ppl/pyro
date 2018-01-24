from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.models import GPRegression
from tests.common import assert_equal


def test_forward_gpr():
    kernel = RBF(input_dim=3, variance=torch.ones(1), lengthscale=torch.ones(3))
    X = Variable(torch.Tensor([[1, 2, 3], [4, 5, 6]]))
    y = Variable(torch.Tensor([0, 1]))
    gpr = GPRegression(X, y, kernel, noise=torch.zeros(1))
    Z = X
    loc, cov = gpr(Z)
    assert loc.dim() == 1
    assert cov.dim() == 2
    assert loc.size(0) == 2
    assert cov.size(0) == 2
    assert cov.size(1) == 2
    assert_equal(loc.data.sum(), kernel(X).matmul(y).data.sum())
    assert_equal(cov.data.abs().sum(), 0)
