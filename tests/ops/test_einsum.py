from __future__ import absolute_import, division, print_function

import opt_einsum
import torch

from pyro.ops._einsum import Deferred, DeferredTensor, shared_intermediates
from tests.common import assert_equal


def test_deferred_backend():
    w = torch.randn(2, 3, 4)
    x = torch.randn(3, 4, 5)
    y = torch.randn(4, 5, 6)
    z = torch.randn(5, 6, 7)
    expr = 'abc,bcd,cde,def->af'

    expected = opt_einsum.contract(expr, w, x, y, z, backend='torch')

    with shared_intermediates():
        w_ = DeferredTensor(w)
        x_ = DeferredTensor(x)
        y_ = DeferredTensor(y)
        z_ = DeferredTensor(z)
        actual_ = opt_einsum.contract(expr, w_, x_, y_, z_, backend='pyro.ops._einsum')

    assert isinstance(actual_, Deferred)
    actual = actual_.eval()
    assert_equal(actual, expected)
