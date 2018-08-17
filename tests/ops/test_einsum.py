from __future__ import absolute_import, division, print_function

import opt_einsum
import pytest
import torch

from pyro.ops._einsum import Deferred, deferred_tensor, shared_intermediates
from tests.common import assert_equal


def test_deferred_backend():
    w = torch.randn(2, 3, 4)
    x = torch.randn(3, 4, 5)
    y = torch.randn(4, 5, 6)
    z = torch.randn(5, 6, 7)
    expr = 'abc,bcd,cde,def->af'

    expected = opt_einsum.contract(expr, w, x, y, z, backend='torch')

    with shared_intermediates():
        w_ = deferred_tensor(w)
        x_ = deferred_tensor(x)
        y_ = deferred_tensor(y)
        z_ = deferred_tensor(z)
        actual_ = opt_einsum.contract(expr, w_, x_, y_, z_, backend='pyro.ops._einsum')

    assert isinstance(actual_, Deferred)
    actual = actual_.eval()
    assert_equal(actual, expected)


def test_complete_sharing():
    x = torch.randn(5, 4)
    y = torch.randn(4, 3)
    z = torch.randn(3, 2)

    print('-' * 40)
    print('Without sharing:')
    with shared_intermediates() as cache:
        x_ = deferred_tensor(x)
        y_ = deferred_tensor(y)
        z_ = deferred_tensor(z)
        opt_einsum.contract('ab,bc,cd->', x_, y_, z_, backend='pyro.ops._einsum')
        expected = len(cache)

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        x_ = deferred_tensor(x)
        y_ = deferred_tensor(y)
        z_ = deferred_tensor(z)
        opt_einsum.contract('ab,bc,cd->', x_, y_, z_, backend='pyro.ops._einsum')
        opt_einsum.contract('ab,bc,cd->', x_, y_, z_, backend='pyro.ops._einsum')
        actual = len(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(expected))
    print('With sharing: {} expressions'.format(actual))
    assert actual == expected


def test_partial_sharing():
    x = torch.randn(5, 4)
    y = torch.randn(4, 3)
    z1 = torch.randn(3, 2)
    z2 = torch.randn(3, 2)

    print('-' * 40)
    print('Without sharing:')
    num_exprs_nosharing = 0
    with shared_intermediates() as cache:
        x_ = deferred_tensor(x)
        y_ = deferred_tensor(y)
        z1_ = deferred_tensor(z1)
        opt_einsum.contract('ab,bc,cd->', x_, y_, z1_, backend='pyro.ops._einsum')
        num_exprs_nosharing += len(cache) - 3  # ignore deferred_tensor
    with shared_intermediates() as cache:
        x_ = deferred_tensor(x)
        y_ = deferred_tensor(y)
        z2_ = deferred_tensor(z1)
        opt_einsum.contract('ab,bc,cd->', x_, y_, z2_, backend='pyro.ops._einsum')
        num_exprs_nosharing += len(cache) - 3  # ignore deferred_tensor

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        x_ = deferred_tensor(x)
        y_ = deferred_tensor(y)
        z1_ = deferred_tensor(z1)
        z2_ = deferred_tensor(z2)
        opt_einsum.contract('ab,bc,cd->', x_, y_, z1_, backend='pyro.ops._einsum')
        opt_einsum.contract('ab,bc,cd->', x_, y_, z2_, backend='pyro.ops._einsum')
        num_exprs_sharing = len(cache) - 4  # ignore deferred_tensor

    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing > num_exprs_sharing


def compute_cost(cache):
    return sum(1 for v in cache.values()
               if type(v).__name__ in ('Einsum', 'Tensordot'))


@pytest.mark.parametrize('size', [3, 4, 5])
def test_chain_sharing(size):
    xs = [torch.randn(2, 2) for _ in range(size)]
    alphabet = ''.join(opt_einsum.get_symbol(i) for i in range(size + 1))
    names = [alphabet[i:i+2] for i in range(size)]
    inputs = ','.join(names)

    num_exprs_nosharing = 0
    for i in range(size + 1):
        with shared_intermediates() as cache:
            target = alphabet[i]
            equation = '{}->{}'.format(inputs, target)
            xs_ = [deferred_tensor(x) for x in xs]
            opt_einsum.contract(equation, *xs_, backend='pyro.ops._einsum')
            num_exprs_nosharing += compute_cost(cache)

    with shared_intermediates() as cache:
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            equation = '{}->{}'.format(inputs, target)
            xs_ = [deferred_tensor(x) for x in xs]
            path_info = opt_einsum.contract_path(equation, *xs_)
            print(path_info[1])
            opt_einsum.contract(equation, *xs_, backend='pyro.ops._einsum')
        num_exprs_sharing = compute_cost(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing > num_exprs_sharing
