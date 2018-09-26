from __future__ import absolute_import, division, print_function

import numbers
from collections import OrderedDict

import opt_einsum
import pytest
import torch

from pyro.distributions.util import logsumexp
from pyro.ops.contract import UnpackedLogRing, _partition_terms, contract_tensor_tree, contract_to_tensor, ubersum
from pyro.poutine.indep_messenger import CondIndepStackFrame
from tests.common import assert_equal


def deep_copy(x):
    """
    Deep copy to detect mutation, assuming tensors will not be mutated.
    """
    if isinstance(x, (tuple, frozenset, numbers.Number, torch.Tensor)):
        return x  # assume x is immutable
    if isinstance(x, (list, set)):
        return type(x)(deep_copy(value) for value in x)
    if isinstance(x, (dict, OrderedDict)):
        return type(x)((deep_copy(key), deep_copy(value)) for key, value in x.items())
    raise TypeError(type(x))


def deep_equal(x, y):
    """
    Deep comparison, assuming tensors will not be mutated.
    """
    if type(x) != type(y):
        return False
    if isinstance(x, (tuple, frozenset, set, numbers.Number)):
        return x == y
    if isinstance(x, torch.Tensor):
        return x is y
    if isinstance(x, list):
        if len(x) != len(y):
            return False
        return all((deep_equal(xi, yi) for xi, yi in zip(x, y)))
    if isinstance(x, (dict, OrderedDict)):
        if len(x) != len(y):
            return False
        if any(key not in y for key in x):
            return False
        return all(deep_equal(x[key], y[key]) for key in x)
    raise TypeError(type(x))


def assert_immutable(fn):
    """
    Decorator to check that function args are not mutated.
    """

    def checked_fn(*args):
        copies = tuple(deep_copy(arg) for arg in args)
        result = fn(*args)
        for pos, (arg, copy) in enumerate(zip(args, copies)):
            if not deep_equal(arg, copy):
                raise AssertionError('{} mutated arg {} of type {}.\nOld:\n{}\nNew:\n{}'
                                     .format(fn.__name__, pos, type(arg).__name__, copy, arg))
        return result

    return checked_fn


@pytest.mark.parametrize('shapes,dims,expected_num_components', [
    ([()], set(), 1),
    ([(2,)], set(), 1),
    ([(2,)], set([-1]), 1),
    ([(2,), (2,)], set(), 2),
    ([(2,), (2,)], set([-1]), 1),
    ([(2, 1), (2, 1), (1, 3), (1, 3)], set(), 4),
    ([(2, 1), (2, 1), (1, 3), (1, 3)], set([-1]), 3),
    ([(2, 1), (2, 1), (1, 3), (1, 3)], set([-2]), 3),
    ([(2, 1), (2, 1), (1, 3), (1, 3)], set([-1, -2]), 2),
    ([(2, 1), (2, 3), (1, 3)], set(), 3),
    ([(2, 1), (2, 3), (1, 3)], set([-1]), 2),
    ([(2, 1), (2, 3), (1, 3)], set([-2]), 2),
    ([(2, 1), (2, 3), (1, 3)], set([-1, -2]), 1),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set(), 4),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-1]), 3),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-2]), 3),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-3]), 3),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-1, -3]), 2),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-1, -2, -3]), 1),
])
def test_partition_terms_unpacked(shapes, dims, expected_num_components):
    ring = UnpackedLogRing()
    tensors = [torch.randn(shape) for shape in shapes]
    components = list(_partition_terms(ring, tensors, dims))

    # Check that result is a partition.
    expected_terms = sorted(tensors, key=id)
    actual_terms = sorted((x for c in components for x in c[0]), key=id)
    assert actual_terms == expected_terms
    assert dims == set.union(set(), *(c[1] for c in components))

    # Check that partition is not too coarse.
    assert len(components) == expected_num_components

    # Check that partition is not too fine.
    component_dict = {x: i for i, (terms, _) in enumerate(components) for x in terms}
    for x in tensors:
        for y in tensors:
            if x is not y:
                if any(dim >= -x.dim() and x.shape[dim] > 1 and
                       dim >= -y.dim() and y.shape[dim] > 1 for dim in dims):
                    assert component_dict[x] == component_dict[y]


def frame(dim, size):
    return CondIndepStackFrame(name="iarange_{}".format(size), dim=dim, size=size, counter=0)


EXAMPLES = [
    {
        'shape_tree': {
            (): [(2, 3, 1)],
            (frame(-1, 4),): [(2, 3, 4)],
        },
        'sum_dims': [],
        'target_ordinal': (),
        'expected_shape': (2, 3, 1),
    },
    {
        'shape_tree': {
            (): [(2, 3, 1)],
            (frame(-1, 4),): [(2, 3, 4)],
        },
        'sum_dims': [],
        'target_ordinal': (frame(-1, 4),),
        'expected_shape': (2, 3, 4),
    },
    # ------------------------------------------------------
    #          z
    #          | 4    max_iarange_nesting=2
    #    x     y      w, x, y, z are all enumerated in dims:
    #   2 \   / 3    -3 -4 -5 -6
    #       w
    {
        'shape_tree': {
            (): [(2, 1, 1)],  # w
            (frame(-1, 2),): [(2, 2, 1, 2)],  # x
            (frame(-1, 3),): [(2, 1, 2, 1, 3)],  # y
            (frame(-1, 3), frame(-2, 4)): [(2, 2, 1, 1, 4, 3)],  # z
        },
        # query for w
        'sum_dims': [-4, -5, -6],
        'target_ordinal': (),
        'expected_shape': (2, 1, 1),
    },
    {
        'shape_tree': {
            (): [(2, 1, 1)],  # w
            (frame(-1, 2),): [(2, 2, 1, 2)],  # x
            (frame(-1, 3),): [(2, 1, 2, 1, 3)],  # y
            (frame(-1, 3), frame(-2, 4)): [(2, 2, 1, 1, 4, 3)],  # z
        },
        # query for x
        'sum_dims': [-3, -5, -6],
        'target_ordinal': (frame(-1, 2),),
        'expected_shape': (2, 1, 1, 2),
    },
    {
        'shape_tree': {
            (): [(2, 1, 1)],  # w
            (frame(-1, 2),): [(2, 2, 1, 2)],  # x
            (frame(-1, 3),): [(2, 1, 2, 1, 3)],  # y
            (frame(-1, 3), frame(-2, 4)): [(2, 2, 1, 1, 4, 3)],  # z
        },
        # query for x
        'sum_dims': [-3, -4, -6],
        'target_ordinal': (frame(-1, 3),),
        'expected_shape': (2, 1, 1, 1, 3),
    },
    {
        'shape_tree': {
            (): [(2, 1, 1)],  # w
            (frame(-1, 2),): [(2, 2, 1, 2)],  # x
            (frame(-1, 3),): [(2, 1, 2, 1, 3)],  # y
            (frame(-1, 3), frame(-2, 4)): [(2, 2, 1, 1, 4, 3)],  # z
        },
        # query for z
        'sum_dims': [-3, -4, -5],
        'target_ordinal': (frame(-1, 3), frame(-2, 4)),
        'expected_shape': (2, 1, 1, 1, 4, 3),
    },
]


@pytest.mark.parametrize('example', EXAMPLES)
def test_contract_to_tensor(example):
    tensor_tree = OrderedDict((frozenset(t), [torch.randn(shape) for shape in shapes])
                              for t, shapes in example['shape_tree'].items())
    sum_dims = {x: set(d for d in example['sum_dims'] if -d <= x.dim() and x.shape[d] > 1)
                for terms in tensor_tree.values()
                for x in terms}
    target_ordinal = frozenset(example['target_ordinal'])
    expected_shape = example['expected_shape']

    actual = assert_immutable(contract_to_tensor)(tensor_tree, sum_dims, target_ordinal)
    assert actual.shape == expected_shape


@pytest.mark.parametrize('example', EXAMPLES)
def test_contract_tensor_tree(example):
    tensor_tree = OrderedDict((frozenset(t), [torch.randn(shape) for shape in shapes])
                              for t, shapes in example['shape_tree'].items())
    sum_dims = {x: set(d for d in example['sum_dims'] if -d <= x.dim() and x.shape[d] > 1)
                for terms in tensor_tree.values()
                for x in terms}

    actual = assert_immutable(contract_tensor_tree)(tensor_tree, sum_dims)
    assert actual
    for ordinal, terms in actual.items():
        for term in terms:
            for frame in ordinal:
                assert term.shape[frame.dim] == frame.size


@pytest.mark.parametrize('a', [2, 1])
@pytest.mark.parametrize('b', [3, 1])
@pytest.mark.parametrize('c', [3, 1])
@pytest.mark.parametrize('d', [4, 1])
def test_contract_to_tensor_sizes(a, b, c, d):
    X = torch.randn(a, 1, c, 1)
    Y = torch.randn(a, b, 1, 1)
    Z = torch.randn(b, 1, d)
    tensor_tree = OrderedDict([(frozenset([frame(-2, c)]), [X]),
                               (frozenset(), [Y]),
                               (frozenset([frame(-1, d)]), [Z])])
    sum_dims = {X: {-4}, Y: {-4, -3}, Z: {-3}}

    target_ordinal = frozenset()
    actual = contract_to_tensor(tensor_tree, sum_dims, target_ordinal)
    assert actual.shape == ()

    target_ordinal = frozenset([frame(-2, c)])
    actual = contract_to_tensor(tensor_tree, sum_dims, target_ordinal)
    assert actual.shape == (c, 1)

    target_ordinal = frozenset([frame(-1, d)])
    actual = contract_to_tensor(tensor_tree, sum_dims, target_ordinal)
    assert actual.shape == (d,)


UBERSUM_EXAMPLES = [
    ('->', ''),
    ('a->', ''),
    ('ab->', ''),
    ('ab,bc->ac', ''),
    ('ab,bc->ca', ''),
    ('ab,bc->a,b,c', ''),
    ('ab,bc,cd->cb,da', ''),
    ('ab,ac->,b,c,cb,a,ca,ba,ab,ac,bac', 'a'),
    ('e,ae,be,bce,bde->,e,a,ae,b,be,bc,bce,bd,bde', 'abcd'),
]


@pytest.mark.parametrize('equation,batch_dims', UBERSUM_EXAMPLES)
def test_ubersum(equation, batch_dims):
    symbols = sorted(set(equation) - set(',->'))
    sizes = {dim: size for dim, size in zip(symbols, range(2, 2 + len(symbols)))}
    inputs, outputs = equation.split('->')
    operands = []
    for dims in inputs.split(','):
        shape = tuple(sizes[dim] for dim in dims)
        operands.append(torch.randn(shape))

    actual = ubersum(equation, *operands, batch_dims=batch_dims)

    outputs = outputs.split(',')
    assert len(actual) == len(outputs)
    for output, actual_part in zip(outputs, actual):
        expected_shape = tuple(sizes[dim] for dim in output)
        assert actual_part.shape == expected_shape
        if set(batch_dims) <= set(output):
            equation_part = inputs + '->' + output
            expected_part = opt_einsum.contract(equation_part, *operands,
                                                backend='pyro.ops.einsum.torch_log')
            assert_equal(actual_part, expected_part)


@pytest.mark.parametrize('a', [2, 1])
@pytest.mark.parametrize('b', [3, 1])
@pytest.mark.parametrize('c', [3, 1])
@pytest.mark.parametrize('d', [4, 1])
def test_ubersum_sizes(a, b, c, d):
    X = torch.randn(a, b)
    Y = torch.randn(b, c)
    Z = torch.randn(c, d)
    actual = ubersum('ab,bc,cd->a,b,c,d', X, Y, Z, batch_dims='ad')
    actual_a, actual_b, actual_c, actual_d = actual
    assert actual_a.shape == (a,)
    assert actual_b.shape == (b,)
    assert actual_c.shape == (c,)
    assert actual_d.shape == (d,)


def test_ubersum_1():
    # y {a}   z {b}
    #      \  /
    #     x {}  <--- target
    a, b, c, d, e = 2, 3, 4, 5, 6
    x = torch.randn(c)
    y = torch.randn(c, d, a)
    z = torch.randn(e, c, b)
    actual, = ubersum('c,cda,ecb->', x, y, z, batch_dims='ab')
    expected = logsumexp(x + logsumexp(y, -2).sum(-1) + logsumexp(z, -3).sum(-1), -1)
    assert_equal(actual, expected)


def test_ubersum_2():
    # y {a}   z {b}  <--- target
    #      \  /
    #     x {}
    a, b, c, d, e = 2, 3, 4, 5, 6
    x = torch.randn(c)
    y = torch.randn(c, d, a)
    z = torch.randn(e, c, b)
    actual, = ubersum('c,cda,ecb->b', x, y, z, batch_dims='ab')
    xyz = logsumexp(x + logsumexp(y, -2).sum(-1) + logsumexp(z, -3).sum(-1), -1)
    expected = xyz.expand(b)
    assert_equal(actual, expected)


def test_ubersum_3():
    #       z {b,c}
    #           |
    # w {a}  y {b}  <--- target
    #      \  /
    #     x {}
    a, b, c, d, e = 2, 3, 4, 5, 6
    w = torch.randn(a, e)
    x = torch.randn(d)
    y = torch.randn(b, d)
    z = torch.randn(b, c, d, e)
    actual, = ubersum('ae,d,bd,bcde->be', w, x, y, z, batch_dims='abc')
    yz = y.reshape(b, d, 1) + z.sum(-3)  # eliminate c
    assert yz.shape == (b, d, e)
    yz = yz.sum(0)  # eliminate b
    assert yz.shape == (d, e)
    wxyz = w.sum(0) + x.reshape(d, 1) + yz  # eliminate a
    assert wxyz.shape == (d, e)
    wxyz = logsumexp(wxyz, 0)  # eliminate d
    assert wxyz.shape == (e,)
    expected = wxyz.expand(b, e)  # broadcast to b
    assert_equal(actual, expected)


def test_ubersum_collide_error():
    # Non-tree iaranges cause exponential blowup,
    # so ubersum() refuses to evaluate them.
    #
    #   z {a,b}
    #     /   \
    # x {a}  y {b}
    #      \  /
    #       {}  <--- target
    a, b, c, d = 2, 3, 4, 5
    x = torch.randn(a, c)
    y = torch.randn(b, d)
    z = torch.randn(a, b, c, d)
    with pytest.raises(ValueError, match='Expected tree-structured iarange nesting'):
        ubersum('ac,bd,abcd->', x, y, z, batch_dims='ab')


def test_ubersum_collide_ok_1():
    # The following is ok because it splits into connected components
    # {x,z1} and {y,z2}, thereby avoiding exponential blowup.
    #
    # z1,z2 {a,b}
    #       /   \
    #   x {a}  y {b}
    #        \  /
    #         {}  <--- target
    a, b, c, d = 2, 3, 4, 5
    x = torch.randn(a, c)
    y = torch.randn(b, d)
    z1 = torch.randn(a, b, c)
    z2 = torch.randn(a, b, d)
    ubersum('ac,bd,abc,abd->', x, y, z1, z2, batch_dims='ab')


def test_ubersum_collide_ok_2():
    # The following is ok because z1 can be contracted to x and
    # z2 can be contracted to y.
    #
    # z1,z2 {a,b}
    #       /   \
    #   x {a}  y {b}
    #        \  /
    #       w {}  <--- target
    a, b, c, d = 2, 3, 4, 5
    w = torch.randn(c, d)
    x = torch.randn(a, c)
    y = torch.randn(b, d)
    z1 = torch.randn(a, b, c)
    z2 = torch.randn(a, b, d)
    ubersum('cd,ac,bd,abc,abd->', w, x, y, z1, z2, batch_dims='ab')


def test_ubersum_collide_ok_3():
    # The following is ok because x, y, and z can be independently contracted to w.
    #
    #      z {a,b}
    # x {a}   |   y {b}
    #      \  |  /
    #       \ | /
    #       w {}  <--- target
    a, b, c = 2, 3, 4
    w = torch.randn(c)
    x = torch.randn(a, c)
    y = torch.randn(b, c)
    z = torch.randn(a, b, c)
    ubersum('c,ac,bc,abc->', w, x, y, z, batch_dims='ab')


UBERSUM_ERRORS = [
    ('ab,bc->', [(2, 3), (4, 5)], ''),
    ('ab,bc->', [(2, 3), (4, 5)], 'b'),
]


@pytest.mark.parametrize('equation,shapes,batch_dims', UBERSUM_ERRORS)
def test_ubersum_size_error(equation, shapes, batch_dims):
    operands = [torch.randn(shape) for shape in shapes]
    with pytest.raises(ValueError, match='Dimension size mismatch'):
        ubersum(equation, *operands, batch_dims=batch_dims)
