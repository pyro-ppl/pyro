from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.contrib.gp.kernels import RBF
from tests.common import assert_equal


def test_forward_rbf():
    kernel = RBF(input_dim=3, variance=torch.Tensor([2]), lengthscale=torch.Tensor([2, 2, 2]))
    X = Variable(torch.Tensor([[1, 0, 1], [2, 1, 3]]))
    Z = Variable(torch.Tensor([[4, 5, 6], [3, 1, 7]]))
    K = kernel(X, Z)

    assert K.dim() == 2
    assert K.size(0) == 2
    assert K.size(1) == 2
    assert_equal(K.data.sum(), 0.30531)


def test_Kdiag():
    kernel = RBF(input_dim=3, variance=torch.Tensor([3]), lengthscale=torch.Tensor([2, 1, 2]))
    X = Variable(torch.Tensor([[1, 2, 1], [2, 3, 3]]))
    K = kernel(X)
    Kdiag = kernel(X, diag=True)

    assert_equal(K.diag(), Kdiag)
