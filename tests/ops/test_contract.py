# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
import numbers
from collections import OrderedDict

import opt_einsum
import pytest
import torch

import pyro.ops.jit
from pyro.distributions.util import logsumexp
from pyro.ops.contract import _partition_terms, contract_tensor_tree, contract_to_tensor, einsum, naive_ubersum, ubersum
from pyro.ops.einsum.adjoint import require_backward
from pyro.ops.rings import LogRing
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
    if isinstance(x, str):
        return x
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


def _normalize(tensor, dims, plates):
    total = tensor
    for i, dim in enumerate(dims):
        if dim not in plates:
            total = logsumexp(total, i, keepdim=True)
    return tensor - total


@pytest.mark.parametrize('inputs,dims,expected_num_components', [
    ([''], set(), 1),
    (['a'], set(), 1),
    (['a'], set('a'), 1),
    (['a', 'a'], set(), 2),
    (['a', 'a'], set('a'), 1),
    (['a', 'a', 'b', 'b'], set(), 4),
    (['a', 'a', 'b', 'b'], set('a'), 3),
    (['a', 'a', 'b', 'b'], set('b'), 3),
    (['a', 'a', 'b', 'b'], set('ab'), 2),
    (['a', 'ab', 'b'], set(), 3),
    (['a', 'ab', 'b'], set('a'), 2),
    (['a', 'ab', 'b'], set('b'), 2),
    (['a', 'ab', 'b'], set('ab'), 1),
    (['a', 'ab', 'bc', 'c'], set(), 4),
    (['a', 'ab', 'bc', 'c'], set('c'), 3),
    (['a', 'ab', 'bc', 'c'], set('b'), 3),
    (['a', 'ab', 'bc', 'c'], set('a'), 3),
    (['a', 'ab', 'bc', 'c'], set('ac'), 2),
    (['a', 'ab', 'bc', 'c'], set('abc'), 1),
])
def test_partition_terms(inputs, dims, expected_num_components):
    ring = LogRing()
    symbol_to_size = dict(zip('abc', [2, 3, 4]))
    shapes = [tuple(symbol_to_size[s] for s in input_) for input_ in inputs]
    tensors = [torch.randn(shape) for shape in shapes]
    for input_, tensor in zip(inputs, tensors):
        tensor._pyro_dims = input_
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
                if dims.intersection(x._pyro_dims, y._pyro_dims):
                    assert component_dict[x] == component_dict[y]


def frame(dim, size):
    return CondIndepStackFrame(name="plate_{}".format(size), dim=dim, size=size, counter=0)


EXAMPLES = [
    # ------------------------------------------------------
    #  y      max_plate_nesting=1
    #  | 4    x, y are enumerated in dims:
    #  x      a, b
    {
        'shape_tree': {
            frozenset(): ['a'],
            frozenset('i'): ['abi'],
        },
        'sum_dims': set('ab'),
        'target_dims': set(),
        'target_ordinal': frozenset(),
        'expected_dims': (),
    },
    {
        'shape_tree': {
            frozenset(): ['a'],
            frozenset('i'): ['abi'],
        },
        'sum_dims': set('ab'),
        'target_dims': set('a'),
        'target_ordinal': frozenset(),
        'expected_dims': 'a',
    },
    {
        'shape_tree': {
            frozenset(): ['a'],
            frozenset('i'): ['abi'],
        },
        'sum_dims': set('ab'),
        'target_dims': set('b'),
        'target_ordinal': frozenset('i'),
        'expected_dims': 'bi',
    },
    {
        'shape_tree': {
            frozenset(): ['a'],
            frozenset('i'): ['abi'],
        },
        'sum_dims': set('ab'),
        'target_dims': set('ab'),
        'target_ordinal': frozenset('i'),
        'expected_dims': 'abi',
    },
    # ------------------------------------------------------
    #          z
    #          | 4    max_plate_nesting=2
    #    x     y      w, x, y, z are all enumerated in dims:
    #   2 \   / 3     a, b, c, d
    #       w
    {
        'shape_tree': {
            frozenset(): ['a'],  # w
            frozenset('i'): ['abi'],  # x
            frozenset('j'): ['acj'],  # y
            frozenset('ij'): ['cdij'],  # z
        },
        # query for w
        'sum_dims': set('abcd'),
        'target_dims': set('a'),
        'target_ordinal': frozenset(),
        'expected_dims': 'a',
    },
    {
        'shape_tree': {
            frozenset(): ['a'],  # w
            frozenset('i'): ['abi'],  # x
            frozenset('j'): ['acj'],  # y
            frozenset('ij'): ['cdij'],  # z
        },
        # query for x
        'sum_dims': set('abcd'),
        'target_dims': set('b'),
        'target_ordinal': frozenset('i'),
        'expected_dims': 'bi',
    },
    {
        'shape_tree': {
            frozenset(): ['a'],  # w
            frozenset('i'): ['abi'],  # x
            frozenset('j'): ['acj'],  # y
            frozenset('ij'): ['cdij'],  # z
        },
        # query for y
        'sum_dims': set('abcd'),
        'target_dims': set('c'),
        'target_ordinal': frozenset('j'),
        'expected_dims': 'cj',
    },
    {
        'shape_tree': {
            frozenset(): ['a'],  # w
            frozenset('i'): ['abi'],  # x
            frozenset('j'): ['acj'],  # y
            frozenset('ij'): ['cdij'],  # z
        },
        # query for z
        'sum_dims': set('abcd'),
        'target_dims': set('d'),
        'target_ordinal': frozenset('ij'),
        'expected_dims': 'dij',
    },
]


@pytest.mark.parametrize('example', EXAMPLES)
def test_contract_to_tensor(example):
    symbol_to_size = dict(zip('abcdij', [4, 5, 6, 7, 2, 3]))
    tensor_tree = OrderedDict()
    for t, shapes in example['shape_tree'].items():
        for dims in shapes:
            tensor = torch.randn(tuple(symbol_to_size[s] for s in dims))
            tensor._pyro_dims = dims
            tensor_tree.setdefault(t, []).append(tensor)
    sum_dims = example['sum_dims']
    target_dims = example['target_dims']
    target_ordinal = example['target_ordinal']
    expected_dims = example['expected_dims']

    actual = assert_immutable(contract_to_tensor)(tensor_tree, sum_dims, target_ordinal, target_dims)
    assert set(actual._pyro_dims) == set(expected_dims)


@pytest.mark.parametrize('example', EXAMPLES)
def test_contract_tensor_tree(example):
    symbol_to_size = dict(zip('abcdij', [4, 5, 6, 7, 2, 3]))
    tensor_tree = OrderedDict()
    for t, shapes in example['shape_tree'].items():
        for dims in shapes:
            tensor = torch.randn(tuple(symbol_to_size[s] for s in dims))
            tensor._pyro_dims = dims
            tensor_tree.setdefault(t, []).append(tensor)
    sum_dims = example['sum_dims']

    tensor_tree = assert_immutable(contract_tensor_tree)(tensor_tree, sum_dims)
    assert tensor_tree
    for ordinal, terms in tensor_tree.items():
        for term in terms:
            for frame in ordinal:
                assert term.shape[frame.dim] == frame.size


# Let abcde be enum dims and ijk be plates.
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
    ('a,b,c,di,ei,fj->,a,b,c,di,ei,fj', 'ij'),
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
        operands.append(torch.randn(shape) if fill is None else torch.full(shape, fill))
    return inputs, outputs, operands, sizes


@pytest.mark.parametrize('equation,plates', UBERSUM_EXAMPLES)
def test_naive_ubersum(equation, plates):
    inputs, outputs, operands, sizes = make_example(equation)

    actual = naive_ubersum(equation, *operands, plates=plates)

    assert isinstance(actual, tuple)
    assert len(actual) == len(outputs)
    for output, actual_part in zip(outputs, actual):
        expected_shape = tuple(sizes[dim] for dim in output)
        assert actual_part.shape == expected_shape
        if not plates:
            equation_part = ','.join(inputs) + '->' + output
            expected_part = opt_einsum.contract(equation_part, *operands,
                                                backend='pyro.ops.einsum.torch_log')
            assert_equal(expected_part, actual_part,
                         msg=u"For output '{}':\nExpected:\n{}\nActual:\n{}".format(
                             output, expected_part.detach().cpu(), actual_part.detach().cpu()))


@pytest.mark.parametrize('equation,plates', UBERSUM_EXAMPLES)
def test_ubersum(equation, plates):
    inputs, outputs, operands, sizes = make_example(equation)

    try:
        actual = ubersum(equation, *operands, plates=plates, modulo_total=True)
    except NotImplementedError:
        pytest.skip()

    assert isinstance(actual, tuple)
    assert len(actual) == len(outputs)
    expected = naive_ubersum(equation, *operands, plates=plates)
    for output, expected_part, actual_part in zip(outputs, expected, actual):
        actual_part = _normalize(actual_part, output, plates)
        expected_part = _normalize(expected_part, output, plates)
        assert_equal(expected_part, actual_part,
                     msg=u"For output '{}':\nExpected:\n{}\nActual:\n{}".format(
                         output, expected_part.detach().cpu(), actual_part.detach().cpu()))


@pytest.mark.parametrize('equation,plates', UBERSUM_EXAMPLES)
def test_einsum_linear(equation, plates):
    inputs, outputs, log_operands, sizes = make_example(equation)
    operands = [x.exp() for x in log_operands]

    try:
        log_expected = ubersum(equation, *log_operands, plates=plates, modulo_total=True)
        expected = [x.exp() for x in log_expected]
    except NotImplementedError:
        pytest.skip()

    # einsum() is in linear space whereas ubersum() is in log space.
    actual = einsum(equation, *operands, plates=plates, modulo_total=True)
    assert isinstance(actual, tuple)
    assert len(actual) == len(outputs)
    for output, expected_part, actual_part in zip(outputs, expected, actual):
        assert_equal(expected_part.log(), actual_part.log(),
                     msg=u"For output '{}':\nExpected:\n{}\nActual:\n{}".format(
                         output, expected_part.detach().cpu(), actual_part.detach().cpu()))


@pytest.mark.parametrize('equation,plates', UBERSUM_EXAMPLES)
def test_ubersum_jit(equation, plates):
    inputs, outputs, operands, sizes = make_example(equation)

    try:
        expected = ubersum(equation, *operands, plates=plates, modulo_total=True)
    except NotImplementedError:
        pytest.skip()

    @pyro.ops.jit.trace
    def jit_ubersum(*operands):
        return ubersum(equation, *operands, plates=plates, modulo_total=True)

    actual = jit_ubersum(*operands)

    if not isinstance(actual, tuple):
        pytest.xfail(reason="https://github.com/pytorch/pytorch/issues/14875")
    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        assert_equal(e, a)


@pytest.mark.parametrize('equation,plates', [
    ('i->', 'i'),
    ('i->i', 'i'),
    (',i->', 'i'),
    (',i->i', 'i'),
    ('ai->', 'i'),
    ('ai->i', 'i'),
    ('ai->ai', 'i'),
    (',ai,abij->aij', 'ij'),
    ('a,ai,bij->bij', 'ij'),
    ('a,ai,abij->bij', 'ij'),
    ('a,abi,bcij->a', 'ij'),
    ('a,abi,bcij->bi', 'ij'),
    ('a,abi,bcij->bij', 'ij'),
    ('a,abi,bcij->cij', 'ij'),
    ('ab,bcdi,deij->eij', 'ij'),
])
def test_ubersum_total(equation, plates):
    inputs, outputs, operands, sizes = make_example(equation, fill=1., sizes=(2,))
    output = outputs[0]

    expected = naive_ubersum(equation, *operands, plates=plates)[0]
    actual = ubersum(equation, *operands, plates=plates, modulo_total=True)[0]
    expected = _normalize(expected, output, plates)
    actual = _normalize(actual, output, plates)
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
    actual = impl('ab,bc,cd->a,b,c,d', X, Y, Z, plates='ad', modulo_total=True)
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
    actual, = impl('c,cda,ecb->', x, y, z, plates='ab', modulo_total=True)
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
    actual, = impl('c,cda,ecb->b', x, y, z, plates='ab', modulo_total=True)
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
    actual, = impl('ae,d,bd,bcde->be', w, x, y, z, plates='abc', modulo_total=True)
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
    actual, = impl('ab,dbc->dc', x, y, plates='d', modulo_total=True)
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
    actual, = impl('a,abi,bcij->cij', x, y, z, plates='ij', modulo_total=True)

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
        impl('ac,bd,abcd->', x, y, z, plates='ab', modulo_total=True)


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
    impl('ac,bd,abc,abd->', x, y, z1, z2, plates='ab', modulo_total=True)


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
    impl('cd,ac,bd,abc,abd->', w, x, y, z1, z2, plates='ab', modulo_total=True)


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
    impl('c,ac,bc,abc->', w, x, y, z, plates='ab', modulo_total=True)


UBERSUM_SHAPE_ERRORS = [
    ('ab,bc->', [(2, 3), (4, 5)], ''),
    ('ab,bc->', [(2, 3), (4, 5)], 'b'),
]


@pytest.mark.parametrize('equation,shapes,plates', UBERSUM_SHAPE_ERRORS)
@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_size_error(impl, equation, shapes, plates):
    operands = [torch.randn(shape) for shape in shapes]
    with pytest.raises(ValueError, match='Dimension size mismatch|Size of label'):
        impl(equation, *operands, plates=plates, modulo_total=True)


UBERSUM_BATCH_ERRORS = [
    ('ai->a', 'i'),
    (',ai->a', 'i'),
    ('bi,abi->b', 'i'),
    (',bi,abi->b', 'i'),
    ('aij->ai', 'ij'),
    ('aij->aj', 'ij'),
]


@pytest.mark.parametrize('equation,plates', UBERSUM_BATCH_ERRORS)
@pytest.mark.parametrize('impl', [naive_ubersum, ubersum])
def test_ubersum_plate_error(impl, equation, plates):
    inputs, outputs = equation.split('->')
    operands = [torch.randn(torch.Size((2,) * len(input_)))
                for input_ in inputs.split(',')]
    with pytest.raises(ValueError, match='It is nonsensical to preserve a plated dim'):
        impl(equation, *operands, plates=plates, modulo_total=True)


ADJOINT_EXAMPLES = [
    ('a->', ''),
    ('a,a->', ''),
    ('ab,bc->', ''),
    ('a,abi->', 'i'),
    ('a,abi,bcij->', 'ij'),
    ('a,abi,bcij,bdik->', 'ijk'),
    ('ai,ai->i', 'i'),
    ('ai,abij->i', 'ij'),
    ('ai,abij,acik->i', 'ijk'),
]


@pytest.mark.parametrize('equation,plates', ADJOINT_EXAMPLES)
@pytest.mark.parametrize('backend', ['map', 'sample', 'marginal'])
def test_adjoint_shape(backend, equation, plates):
    backend = 'pyro.ops.einsum.torch_{}'.format(backend)
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    operands = [torch.randn(torch.Size((2,) * len(input_)))
                for input_ in inputs]
    for input_, x in zip(inputs, operands):
        x._pyro_dims = input_

    # run forward-backward algorithm
    for x in operands:
        require_backward(x)
    result, = ubersum(equation, *operands, plates=plates,
                      modulo_total=True, backend=backend)
    result._pyro_backward()

    for input_, x in zip(inputs, operands):
        backward_result = x._pyro_backward_result
        contract_dims = set(input_) - set(output) - set(plates)
        if contract_dims:
            assert backward_result is not None
        else:
            assert backward_result is None


@pytest.mark.parametrize('equation,plates', ADJOINT_EXAMPLES)
def test_adjoint_marginal(equation, plates):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    operands = [torch.randn(torch.Size((2,) * len(input_)))
                for input_ in inputs]
    for input_, x in zip(inputs, operands):
        x._pyro_dims = input_

    # check forward pass
    for x in operands:
        require_backward(x)
    actual, = ubersum(equation, *operands, plates=plates, modulo_total=True,
                      backend='pyro.ops.einsum.torch_marginal')
    expected, = ubersum(equation, *operands, plates=plates, modulo_total=True,
                        backend='pyro.ops.einsum.torch_log')
    assert_equal(expected, actual)

    # check backward pass
    actual._pyro_backward()
    for input_, operand in zip(inputs, operands):
        marginal_equation = ','.join(inputs) + '->' + input_
        expected, = ubersum(marginal_equation, *operands, plates=plates, modulo_total=True,
                            backend='pyro.ops.einsum.torch_log')
        actual = operand._pyro_backward_result
        assert_equal(expected, actual)
