from __future__ import absolute_import, division, print_function

import numbers
import operator
import timeit

import pytest
import torch
from opt_einsum import shared_intermediates
from six.moves import reduce

import pyro.distributions.torch_patch  # noqa: F401
from pyro.infer.util import torch_exp
from pyro.ops.einsum import contract
from pyro.ops.sumproduct import logsumproductexp, sumproduct, zip_align_right
from tests.common import assert_equal


@pytest.mark.parametrize('xs,ys,expected', [
    (['a', 'b'], [6, 5], [('a', 6), ('b', 5)]),
    (['b'], [6, 5], [('b', 5)]),
    (['a', 'b'], [5], [('b', 5)]),
])
def test_zip_align_right(xs, ys, expected):
    actual = list(zip_align_right(xs, ys))
    assert actual == expected


# pairs of (factor_shapes, target_shape)
EXAMPLES = [
    ([None], ()),
    ([None], (2,)),
    ([None, None], ()),
    ([None, None], (2,)),
    ([()], ()),
    ([()], (2,)),
    ([(), ()], ()),
    ([(), ()], (2,)),
    ([None, ()], ()),
    ([None, ()], (2,)),
    ([None, (), (2,), (3, 1)], ()),
    ([None, (), (2,), (3, 1)], (2,)),
    ([None, (), (2,), (3, 1)], (3, 1)),
    ([None, (), (2,), (3, 1)], (3, 2)),
    ([None, (), (2,), (3, 1)], (4, 3, 1)),
    ([None, (), (2,), (3, 2), (4, 3, 1)], ()),
    ([None, (), (2,), (3, 2), (4, 3, 1)], (3, 1)),
    ([None, (), (2,), (3, 2), (4, 3, 1)], (5, 1, 3, 1)),
    ([None, (), (2,), (3, 2), (4, 3, 1), (5, 4, 1, 1)], ()),
    ([None, (), (2,), (3, 2), (4, 3, 1), (5, 4, 1, 1)], (4, 1, 1)),
    ([(2,), (3, 2), (4, 3, 1), (5, 4, 1, 1), (3, 1), (4, 1, 1), (5, 1, 1, 1)], ()),
    ([(2,), (3, 2), (4, 3, 1), (5, 4, 1, 1), (3, 1), (4, 1, 1), (5, 1, 1, 1)], (2,)),
    ([(2,), (3, 2), (4, 3, 1), (5, 4, 1, 1), (3, 1), (4, 1, 1), (5, 1, 1, 1)], (5, 1, 3, 1)),
    ([(2,), (3, 2), (4, 3, 1), (5, 4, 1, 1), (3, 1), (4, 1, 1), (5, 1, 1, 1)], ()),
    ([(), (100000, 1, 2, 1), (2, 100000, 1, 1, 1)], (100000, 1, 2, 1)),
    ([None, (100000, 1, 2, 1), (2, 100000, 1, 1, 1)], (100000, 1, 2, 1)),
]


def reference_sumproduct(factors, target_shape):
    tensors = [torch.tensor(x) if isinstance(x, numbers.Number) else x for x in factors]
    result = reduce(operator.mul, tensors, 1.)
    while result.dim() > len(target_shape):
        result = result.sum(0)
    while result.dim() < len(target_shape):
        result = result.unsqueeze(0)
    for i, (e, s) in enumerate(zip(result.shape, target_shape)):
        if e > s:
            result = result.sum(i, keepdim=True)
    result = result.expand(target_shape)
    assert result.shape == target_shape
    return result


@pytest.mark.parametrize('factor_shapes,shape', EXAMPLES)
@pytest.mark.parametrize('optimize', [False, True], ids=['naive', 'opt'])
def test_sumproduct(factor_shapes, shape, optimize):
    factors = [float(torch.randn(torch.Size()).exp().item()) if s is None else
               torch.randn(s).exp()
               for s in factor_shapes]
    actual = sumproduct(factors, shape, optimize=optimize)
    expected = reference_sumproduct(factors, shape)
    assert_equal(actual, expected)


@pytest.mark.parametrize('factor_shapes,shape', EXAMPLES)
@pytest.mark.parametrize('optimize', [False, True], ids=['naive', 'opt'])
def test_logsumproductexp(factor_shapes, shape, optimize):
    log_factors = [float(torch.randn(torch.Size()).item()) if s is None else
                   torch.randn(s)
                   for s in factor_shapes]
    actual = logsumproductexp(log_factors, shape, optimize=optimize)
    factors = [torch_exp(x) for x in log_factors]
    expected_exp = reference_sumproduct(factors, shape)
    expected = expected_exp.log()
    assert_equal(actual, expected)


def test_sharing():
    x = torch.randn(2, 3, 1, 1)
    y = torch.randn(3, 4, 1)
    z = torch.randn(4, 5)

    with shared_intermediates() as cache:
        sumproduct([x, y, z])
    cost_once = len(cache)
    del cache
    assert cost_once > 0, 'computation was not shared'

    with shared_intermediates() as cache:
        sumproduct([x, y, z])
        sumproduct([x, y, z])
    cost_twice = len(cache)
    del cache

    assert cost_twice == cost_once, 'computation was not shared'


# See https://github.com/pytorch/pytorch/issues/10661
@pytest.mark.parametrize('backend', ['torch', 'numpy'])
@pytest.mark.parametrize('equation,shapes', [
    ('ac,abc->cb', [(2, 2000), (2, 2, 2000)]),
    ('ba,->ab', [(2000, 2), ()]),
    ('ab->a', [(2, 2000)]),
    ('a,a->', [(2,), (2,)]),
    ('a,->', [(2,), ()]),
    (',->', [(), ()]),
    (',->', [(), ()]),
    ('->', [()]),
])
def test_einsum_speed(equation, shapes, backend):
    operands = [torch.randn(shape) for shape in shapes]

    start_time = timeit.default_timer()
    for _ in range(1000):
        contract(equation, *operands, backend=backend)
    elapsed = timeit.default_timer() - start_time
    print('{} {}: {}'.format(backend, equation, elapsed))
