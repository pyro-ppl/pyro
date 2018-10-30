from __future__ import absolute_import, division, print_function

import itertools
import numbers
from collections import OrderedDict

import opt_einsum
import pytest
import torch

from pyro.distributions.util import logsumexp
from pyro.ops.contract import (UnpackedLogRing, _partition_terms, contract_tensor_tree, contract_to_tensor,
                               naive_ubersum, ubersum)
from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.util import optional
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

    # Check that the partition is not too coarse.
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
    return CondIndepStackFrame(name="plate_{}".format(size), dim=dim, size=size, counter=0)


EXAMPLES = [
    # ------------------------------------------------------
    #  y      max_plate_nesting=1
    #  | 4    x, y are enumerated in dims:
    #  x     -2 -3
    {
        'shape_tree': {
            (): [(3, 1)],
            (frame(-1, 4),): [(2, 3, 4)],
        },
        'sum_dims': {-2, -3},
        'target_dims': set(),
        'target_ordinal': (),
        'expected_shape': (),
    },
    {
        'shape_tree': {
            (): [(3, 1)],
            (frame(-1, 4),): [(2, 3, 4)],
        },
        'sum_dims': {-2, -3},
        'target_dims': {-2},
        'target_ordinal': (),
        'expected_shape': (3, 1),
    },
    {
        'shape_tree': {
            (): [(3, 1)],
            (frame(-1, 4),): [(2, 3, 4)],
        },
        'sum_dims': {-2, -3},
        'target_dims': {-3},
        'target_ordinal': (frame(-1, 4),),
        'expected_shape': (2, 1, 4),
    },
    {
        'shape_tree': {
            (): [(3, 1)],
            (frame(-1, 4),): [(2, 3, 4)],
        },
        'sum_dims': {-2, -3},
        'target_dims': {-2, -3},
        'target_ordinal': (frame(-1, 4),),
        'expected_shape': (2, 3, 4),
    },
    # ------------------------------------------------------
    #          z
    #          | 4    max_plate_nesting=2
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
        'sum_dims': {-3, -4, -5, -6},
        'target_dims': {-3},
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
        'sum_dims': {-3, -4, -5, -6},
        'target_dims': {-4},
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
        'sum_dims': {-3, -4, -5, -6},
        'target_dims': {-5},
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
        'sum_dims': {-3, -4, -5, -6},
        'target_dims': {-6},
        'target_ordinal': (frame(-1, 3), frame(-2, 4)),
        'expected_shape': (2, 1, 1, 1, 4, 3),
    },
]


@pytest.mark.parametrize('example', EXAMPLES)
def test_contract_to_tensor(example):
    tensor_tree = OrderedDict((frozenset(t), [torch.randn(shape) for shape in shapes])
                              for t, shapes in example['shape_tree'].items())
    sum_dims = example['sum_dims']
    target_dims = example['target_dims']
    target_ordinal = frozenset(example['target_ordinal'])
    expected_shape = example['expected_shape']

    actual = assert_immutable(contract_to_tensor)(tensor_tree, sum_dims, target_ordinal, target_dims)
    assert actual.shape == expected_shape


@pytest.mark.parametrize('example', EXAMPLES)
def test_contract_tensor_tree(example):
    tensor_tree = OrderedDict((frozenset(t), [torch.randn(shape) for shape in shapes])
                              for t, shapes in example['shape_tree'].items())
    sum_dims = example['sum_dims']
    target_dims = example['target_dims']
    ring = UnpackedLogRing()

    forward_tree, backward_tree = assert_immutable(contract_tensor_tree)(tensor_tree, sum_dims, target_dims)
    assert forward_tree
    for tree in (forward_tree, backward_tree):
        for ordinal, terms in tree.items():
            for term in terms:
                for frame in ordinal:
                    assert term.shape[frame.dim] == frame.size
                assert set(ring.dims(term)) <= target_dims


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
    sum_dims = {-4, -3}

    target_ordinal = frozenset()
    actual = contract_to_tensor(tensor_tree, sum_dims, target_ordinal)
    assert actual.shape == ()

    target_ordinal = frozenset([frame(-2, c)])
    actual = contract_to_tensor(tensor_tree, sum_dims, target_ordinal)
    assert actual.shape == (c, 1)

    target_ordinal = frozenset([frame(-1, d)])
    actual = contract_to_tensor(tensor_tree, sum_dims, target_ordinal)
    assert actual.shape == (d,)


# Let abcde be enum dims and ijk be batch dims.
UBERSUM_EXAMPLES = [
    ('->', ''),
    ('a->,a', ''),
    ('ab->,a,b,ab,ba', ''),
    ('ab,bc->,a,b,c,ab,bc,ac,abc', ''),
    ('ab,bc,cd->,a,b,c,d,ab,ac,ad,bc,bd,cd,abc,acd,bcd,abcd', ''),
    ('i->,i', 'i'),
    (',i->,i', 'i'),
    (',i,i->,i', 'i'),
    (',i,ia->,i,ia', 'i'),
    (',i,i,ia,ia->,i,ia', 'i'),
    ('bi,ia->,i,ia,ib,iab', 'i'),
    ('abi,b->,b,ai,abi', 'i'),
    ('ia,ja,ija->,a,i,ia,j,ja,ija', 'ij'),
    ('i,jb,ijab->,i,j,jb,ij,ija,ijb,ijab', 'ij'),
    ('ia,jb,ijab->,i,ia,j,jb,ij,ija,ijb,ijab', 'ij'),
    (',i,j,a,ij,ia,ja,ija->,a,i,j,ia,ja,ij,ija', 'ij'),
    # {ij}   {ik}
    #   a\   /a
    #     {i}
    ('ija,ika->,i,j,k,ij,ik,ijk,ia,ija,ika,ijka', 'ijk'),
    # {ij}   {ik}
    #   a\   /a
    #     {i}      {}
    (',ia,ija,ika->,i,j,k,ij,ik,ijk,ia,ija,ika,ijka', 'ijk'),
    #  {i} c
    #   |b
    #  {} a
    ('ab,bci->,a,b,ab,i,ai,bi,ci,abi,bci,abci', 'i'),
    #  {i} cd
    #   |b
    #  {} a
    ('ab,bci,bdi->,a,b,ab,i,ai,bi,ci,abi,bci,bdi,cdi,abci,abdi,abcdi', 'i'),
    #  {ij} c
    #   |b
    #  {} a
    ('ab,bcij->,a,b,ab,i,j,ij,ai,aj,aij,bi,bj,aij,bij,cij,abij,acij,bcij,abcij', 'ij'),
    #  {ij} c
    #   |b
    #  {i} a
    ('abi,bcij->,i,ai,bi,abi,j,ij,aij,bij,cij,abij,bcij,abcij', 'ij'),
    # {ij} e
    #   |d
    #  {i} c
    #   |b
    #  {} a
    ('ab,bcdi,deij->,a,b,ci,di,eij', 'ij'),
    # {ijk} g
    #   |f
    # {ij} e
    #   |d
    #  {i} c
    #   |b
    #  {} a
    ('ab,bcdi,defij,fgijk->,a,b,ci,di,eij,fij,gijk', 'ijk'),
    # {ik}  {ij}   {ij}
    #   a\   /b    /e
    #     {i}    {j}
    #       c\  /d
    #         {}
    ('aik,bij,abci,cd,dej,eij->,ai,bi,ej,aik,bij,eij', 'ijk'),
    # {ij}    {ij}
    #  a|      |d
    #  {i}    {j}
    #    b\  /c
    #      {}
    ('aij,abi,bc,cdj,dij->,bi,cj,aij,dij,adij', 'ij'),
]


def make_example(equation, fill=None, sizes=(2, 3)):
    symbols = sorted(set(equation) - set(',->'))
    sizes = {dim: size for dim, size in zip(symbols, itertools.cycle(sizes))}
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    outputs = outputs.split(',')
    operands = []
    for dims in inputs:
        shape = tuple(sizes[dim] for dim in dims)
        operands.append(torch.randn(shape) if fill is None else torch.empty(shape).fill_(fill))
    return inputs, outputs, operands, sizes


@pytest.mark.parametrize('equation,batch_dims', UBERSUM_EXAMPLES)
def test_naive_ubersum(equation, batch_dims):
    inputs, outputs, operands, sizes = make_example(equation)

    actual = naive_ubersum(equation, *operands, batch_dims=batch_dims)

    assert isinstance(actual, tuple)
    assert len(actual) == len(outputs)
    for output, actual_part in zip(outputs, actual):
        expected_shape = tuple(sizes[dim] for dim in output)
        assert actual_part.shape == expected_shape
        if not batch_dims:
            equation_part = ','.join(inputs) + '->' + output
            expected_part = opt_einsum.contract(equation_part, *operands,
                                                backend='pyro.ops.einsum.torch_log')
            assert_equal(expected_part, actual_part,
                         msg=u"For output '{}':\nExpected:\n{}\nActual:\n{}".format(
                             output, expected_part.detach().cpu(), actual_part.detach().cpu()))


@pytest.mark.parametrize('equation,batch_dims', UBERSUM_EXAMPLES)
def test_ubersum(equation, batch_dims):
    inputs, outputs, operands, sizes = make_example(equation)

    try:
        actual = ubersum(equation, *operands, batch_dims=batch_dims)
    except NotImplementedError:
        pytest.skip()

    assert isinstance(actual, tuple)
    assert len(actual) == len(outputs)
    expected = naive_ubersum(equation, *operands, batch_dims=batch_dims)
    for output, expected_part, actual_part in zip(outputs, expected, actual):
        assert_equal(expected_part, actual_part,
                     msg=u"For output '{}':\nExpected:\n{}\nActual:\n{}".format(
                         output, expected_part.detach().cpu(), actual_part.detach().cpu()))


@pytest.mark.parametrize('equation,batch_dims', [
    ('i->', 'i'),
    ('i->i', 'i'),
    (',i->', 'i'),
    (',i->i', 'i'),
    (',ai,abij->aij', 'ij'),
    ('a,ai,bij->bij', 'ij'),
    ('a,ai,abij->bij', 'ij'),
    ('a,abi,bcij->a', 'ij'),
    ('a,abi,bcij->bi', 'ij'),
    ('a,abi,bcij->bij', 'ij'),
    ('a,abi,bcij->cij', 'ij'),
    ('ab,bcdi,deij->eij', 'ij'),
])
def test_ubersum_total(equation, batch_dims):
    inputs, outputs, operands, sizes = make_example(equation, fill=1, sizes=(2,))

    expected = naive_ubersum(equation, *operands, batch_dims=batch_dims)[0]
    actual = ubersum(equation, *operands, batch_dims=batch_dims)[0]
    assert_equal(expected, actual,
                 msg=u"Expected:\n{}\nActual:\n{}".format(
                     expected.detach().cpu(), actual.detach().cpu()))


@pytest.mark.parametrize('a', [2, 1])
@pytest.mark.parametrize('b', [3, 1])
@pytest.mark.parametrize('c', [3, 1])
@pytest.mark.parametrize('d', [4, 1])
@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_sizes(impl, a, b, c, d):
    X = torch.randn(a, b)
    Y = torch.randn(b, c)
    Z = torch.randn(c, d)
    actual = impl('ab,bc,cd->a,b,c,d', X, Y, Z, batch_dims='ad')
    actual_a, actual_b, actual_c, actual_d = actual
    assert actual_a.shape == (a,)
    assert actual_b.shape == (b,)
    assert actual_c.shape == (c,)
    assert actual_d.shape == (d,)


@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_1(impl):
    # y {a}   z {b}
    #      \  /
    #     x {}  <--- target
    a, b, c, d, e = 2, 3, 4, 5, 6
    x = torch.randn(c)
    y = torch.randn(c, d, a)
    z = torch.randn(e, c, b)
    actual, = impl('c,cda,ecb->', x, y, z, batch_dims='ab')
    expected = logsumexp(x + logsumexp(y, -2).sum(-1) + logsumexp(z, -3).sum(-1), -1)
    assert_equal(actual, expected)


@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_2(impl):
    # y {a}   z {b}  <--- target
    #      \  /
    #     x {}
    a, b, c, d, e = 2, 3, 4, 5, 6
    x = torch.randn(c)
    y = torch.randn(c, d, a)
    z = torch.randn(e, c, b)
    actual, = impl('c,cda,ecb->b', x, y, z, batch_dims='ab')
    xyz = logsumexp(x + logsumexp(y, -2).sum(-1) + logsumexp(z, -3).sum(-1), -1)
    expected = xyz.expand(b)
    assert_equal(actual, expected)


@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_3(impl):
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
    actual, = impl('ae,d,bd,bcde->be', w, x, y, z, batch_dims='abc')
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


@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_4(impl):
    # x,y {b}  <--- target
    #      |
    #     {}
    a, b, c, d = 2, 3, 4, 5
    x = torch.randn(a, b)
    y = torch.randn(d, b, c)
    actual, = impl('ab,dbc->dc', x, y, batch_dims='d')
    x_b1 = logsumexp(x, 0).unsqueeze(-1)
    assert x_b1.shape == (b, 1)
    y_db1 = logsumexp(y, 2, keepdim=True)
    assert y_db1.shape == (d, b, 1)
    y_dbc = y_db1.sum(0) - y_db1 + y  # inclusion-exclusion
    assert y_dbc.shape == (d, b, c)
    xy_dc = logsumexp(x_b1 + y_dbc, 1)
    assert xy_dc.shape == (d, c)
    expected = xy_dc
    assert_equal(actual, expected)


@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_5(impl):
    # z {ij}  <--- target
    #     |
    #  y {i}
    #     |
    #  x {}
    i, j, a, b, c = 2, 3, 6, 5, 4
    x = torch.randn(a)
    y = torch.randn(a, b, i)
    z = torch.randn(b, c, i, j)
    actual, = impl('a,abi,bcij->cij', x, y, z, batch_dims='ij')

    # contract plate j
    s1 = logsumexp(z, 1)
    assert s1.shape == (b, i, j)
    p1 = s1.sum(2)
    assert p1.shape == (b, i)
    q1 = z - s1.unsqueeze(-3)
    assert q1.shape == (b, c, i, j)

    # contract plate i
    x2 = y + p1
    assert x2.shape == (a, b, i)
    s2 = logsumexp(x2, 1)
    assert s2.shape == (a, i)
    p2 = s2.sum(1)
    assert p2.shape == (a,)
    q2 = x2 - s2.unsqueeze(-2)
    assert q2.shape == (a, b, i)

    expected = opt_einsum.contract('a,a,abi,bcij->cij', x, p2, q2, q1,
                                   backend='pyro.ops.einsum.torch_log')
    assert_equal(actual, expected)


@pytest.mark.parametrize('impl,implemented', [(naive_ubersum, True), (ubersum, False)])
def test_ubersum_collide_implemented(impl, implemented):
    # Non-tree plates cause exponential blowup,
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
    raises = pytest.raises(NotImplementedError, match='Expected tree-structured plate nesting')
    with optional(raises, not implemented):
        impl('ac,bd,abcd->', x, y, z, batch_dims='ab')


@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_collide_ok_1(impl):
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
    impl('ac,bd,abc,abd->', x, y, z1, z2, batch_dims='ab')


@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_collide_ok_2(impl):
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
    impl('cd,ac,bd,abc,abd->', w, x, y, z1, z2, batch_dims='ab')


@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_collide_ok_3(impl):
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
    impl('c,ac,bc,abc->', w, x, y, z, batch_dims='ab')


UBERSUM_SHAPE_ERRORS = [
    ('ab,bc->', [(2, 3), (4, 5)], ''),
    ('ab,bc->', [(2, 3), (4, 5)], 'b'),
]


@pytest.mark.parametrize('equation,shapes,batch_dims', UBERSUM_SHAPE_ERRORS)
@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_size_error(impl, equation, shapes, batch_dims):
    operands = [torch.randn(shape) for shape in shapes]
    with pytest.raises(ValueError, match='Dimension size mismatch|Size of label'):
        impl(equation, *operands, batch_dims=batch_dims)


UBERSUM_BATCH_ERRORS = [
    ('ai->a', 'i'),
    (',ai->a', 'i'),
    ('bi,abi->b', 'i'),
    (',bi,abi->b', 'i'),
    ('aij->ai', 'ij'),
    ('aij->aj', 'ij'),
]


@pytest.mark.parametrize('equation,batch_dims', UBERSUM_BATCH_ERRORS)
@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_batch_error(impl, equation, batch_dims):
    inputs, outputs = equation.split('->')
    operands = [torch.randn(torch.Size((2,) * len(input_)))
                for input_ in inputs.split(',')]
    with pytest.raises(ValueError, match='It is nonsensical to preserve a batched dim'):
        impl(equation, *operands, batch_dims=batch_dims)
