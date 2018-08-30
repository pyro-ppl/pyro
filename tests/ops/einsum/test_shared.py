from __future__ import absolute_import, division, print_function

import opt_einsum
import pytest
import torch
from opt_einsum import shared_intermediates

from pyro.ops.einsum import contract
from tests.common import assert_equal


def compute_cost(cache):
    return sum(1 for key in cache.keys() if key[0] in ('einsum', 'tensordot'))


def test_shared_backend():
    w = torch.randn(2, 3, 4)
    x = torch.randn(3, 4, 5)
    y = torch.randn(4, 5, 6)
    z = torch.randn(5, 6, 7)
    expr = 'abc,bcd,cde,def->af'

    expected = contract(expr, w, x, y, z, backend='torch')
    with shared_intermediates():
        actual = contract(expr, w, x, y, z, backend='torch')

    assert_equal(actual, expected)


def test_complete_sharing():
    x = torch.randn(5, 4)
    y = torch.randn(4, 3)
    z = torch.randn(3, 2)

    print('-' * 40)
    print('Without sharing:')
    with shared_intermediates() as cache:
        contract('ab,bc,cd->', x, y, z, backend='torch')
        expected = len(cache)

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        contract('ab,bc,cd->', x, y, z, backend='torch')
        contract('ab,bc,cd->', x, y, z, backend='torch')
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
        contract('ab,bc,cd->', x, y, z1, backend='torch')
        num_exprs_nosharing += len(cache)
    with shared_intermediates() as cache:
        contract('ab,bc,cd->', x, y, z2, backend='torch')
        num_exprs_nosharing += len(cache)

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        contract('ab,bc,cd->', x, y, z1, backend='torch')
        contract('ab,bc,cd->', x, y, z2, backend='torch')
        num_exprs_sharing = len(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing > num_exprs_sharing


@pytest.mark.parametrize('size', [3, 4, 5])
def test_chain(size):
    xs = [torch.randn(2, 2) for _ in range(size)]
    alphabet = ''.join(opt_einsum.get_symbol(i) for i in range(size + 1))
    names = [alphabet[i:i+2] for i in range(size)]
    inputs = ','.join(names)

    with shared_intermediates():
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            equation = '{}->{}'.format(inputs, target)
            path_info = opt_einsum.contract_path(equation, *xs)
            print(path_info[1])
            contract(equation, *xs, backend='torch')
        print('-' * 40)


@pytest.mark.parametrize('size', [3, 4, 5, 10])
def test_chain_2(size):
    xs = [torch.randn(2, 2) for _ in range(size)]
    alphabet = ''.join(opt_einsum.get_symbol(i) for i in range(size + 1))
    names = [alphabet[i:i+2] for i in range(size)]
    inputs = ','.join(names)

    with shared_intermediates():
        print(inputs)
        for i in range(size):
            target = alphabet[i:i+2]
            equation = '{}->{}'.format(inputs, target)
            path_info = opt_einsum.contract_path(equation, *xs)
            print(path_info[1])
            contract(equation, *xs, backend='torch')
        print('-' * 40)


def test_chain_2_growth():
    sizes = list(range(1, 21))
    costs = []
    for size in sizes:
        xs = [torch.randn(2, 2) for _ in range(size)]
        alphabet = ''.join(opt_einsum.get_symbol(i) for i in range(size + 1))
        names = [alphabet[i:i+2] for i in range(size)]
        inputs = ','.join(names)

        with shared_intermediates() as cache:
            for i in range(size):
                target = alphabet[i:i+2]
                equation = '{}->{}'.format(inputs, target)
                contract(equation, *xs, backend='torch')
            costs.append(compute_cost(cache))

    print('sizes = {}'.format(repr(sizes)))
    print('costs = {}'.format(repr(costs)))
    for size, cost in zip(sizes, costs):
        print('{}\t{}'.format(size, cost))


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
            contract(equation, *xs, backend='torch')
            num_exprs_nosharing += compute_cost(cache)

    with shared_intermediates() as cache:
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            equation = '{}->{}'.format(inputs, target)
            path_info = opt_einsum.contract_path(equation, *xs)
            print(path_info[1])
            contract(equation, *xs, backend='torch')
        num_exprs_sharing = compute_cost(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing > num_exprs_sharing
