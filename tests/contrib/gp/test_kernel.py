from __future__ import absolute_import, division, print_function

from pyro.contrib.gp.kernels import RBF
from tests.common import assert_equal

def test_K_rbf():
    kernel = RBF(variance=torch.Tensor([2]), lengthscale=torch.Tensor([2]))
    X = Variable(torch.Tensor([[1, 2], [3, 4]]))
    Z = Variable(torch.Tensor([[5, 6], [7, 8]]))
    K = kernel.K(X, Z)
    assert_equal(K.sum(), 1.36788)
