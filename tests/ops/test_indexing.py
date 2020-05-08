# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro.ops.indexing import Index, Vindex
from tests.common import assert_equal


class TensorMock:
    def __getitem__(self, args):
        return args


tensor_mock = TensorMock()


def z(*args):
    return torch.zeros(*args, dtype=torch.long)


SHAPE_EXAMPLES = [
    ('Vindex(z(()))[...]', ()),
    ('Vindex(z(2))[...]', (2,)),
    ('Vindex(z(2))[...,0]', ()),
    ('Vindex(z(2))[...,:]', (2,)),
    ('Vindex(z(2))[...,z(3)]', (3,)),
    ('Vindex(z(2))[0]', ()),
    ('Vindex(z(2))[:]', (2,)),
    ('Vindex(z(2))[z(3)]', (3,)),
    ('Vindex(z(2,3))[...]', (2, 3)),
    ('Vindex(z(2,3))[...,0]', (2,)),
    ('Vindex(z(2,3))[...,:]', (2, 3)),
    ('Vindex(z(2,3))[...,z(2)]', (2,)),
    ('Vindex(z(2,3))[...,z(4,1)]', (4, 2)),
    ('Vindex(z(2,3))[...,0,0]', ()),
    ('Vindex(z(2,3))[...,0,:]', (3,)),
    ('Vindex(z(2,3))[...,0,z(4)]', (4,)),
    ('Vindex(z(2,3))[...,:,0]', (2,)),
    ('Vindex(z(2,3))[...,:,:]', (2, 3)),
    ('Vindex(z(2,3))[...,:,z(4)]', (4, 2)),
    ('Vindex(z(2,3))[...,z(4),0]', (4,)),
    ('Vindex(z(2,3))[...,z(4),:]', (4, 3)),
    ('Vindex(z(2,3))[...,z(4),z(4)]', (4,)),
    ('Vindex(z(2,3))[...,z(5,1),z(4)]', (5, 4)),
    ('Vindex(z(2,3))[...,z(4),z(5,1)]', (5, 4)),
    ('Vindex(z(2,3))[0,0]', ()),
    ('Vindex(z(2,3))[0,:]', (3,)),
    ('Vindex(z(2,3))[0,z(4)]', (4,)),
    ('Vindex(z(2,3))[:,0]', (2,)),
    ('Vindex(z(2,3))[:,:]', (2, 3)),
    ('Vindex(z(2,3))[:,z(4)]', (4, 2)),
    ('Vindex(z(2,3))[z(4),0]', (4,)),
    ('Vindex(z(2,3))[z(4),:]', (4, 3)),
    ('Vindex(z(2,3))[z(4)]', (4, 3)),
    ('Vindex(z(2,3))[z(4),z(4)]', (4,)),
    ('Vindex(z(2,3))[z(5,1),z(4)]', (5, 4)),
    ('Vindex(z(2,3))[z(4),z(5,1)]', (5, 4)),
    ('Vindex(z(2,3,4))[...]', (2, 3, 4)),
    ('Vindex(z(2,3,4))[...,z(3)]', (2, 3)),
    ('Vindex(z(2,3,4))[...,z(2,1)]', (2, 3)),
    ('Vindex(z(2,3,4))[...,z(2,3)]', (2, 3)),
    ('Vindex(z(2,3,4))[...,z(5,1,1)]', (5, 2, 3)),
    ('Vindex(z(2,3,4))[...,z(2),0]', (2,)),
    ('Vindex(z(2,3,4))[...,z(5,1),0]', (5, 2)),
    ('Vindex(z(2,3,4))[...,z(2),:]', (2, 4)),
    ('Vindex(z(2,3,4))[...,z(5,1),:]', (5, 2, 4)),
    ('Vindex(z(2,3,4))[...,z(5),0,0]', (5,)),
    ('Vindex(z(2,3,4))[...,z(5),0,:]', (5, 4)),
    ('Vindex(z(2,3,4))[...,z(5),:,0]', (5, 3)),
    ('Vindex(z(2,3,4))[...,z(5),:,:]', (5, 3, 4)),
    ('Vindex(z(2,3,4))[0,0,z(5)]', (5,)),
    ('Vindex(z(2,3,4))[0,:,z(5)]', (5, 3)),
    ('Vindex(z(2,3,4))[0,z(5),0]', (5,)),
    ('Vindex(z(2,3,4))[0,z(5),:]', (5, 4)),
    ('Vindex(z(2,3,4))[0,z(5),z(5)]', (5,)),
    ('Vindex(z(2,3,4))[0,z(5,1),z(6)]', (5, 6)),
    ('Vindex(z(2,3,4))[0,z(6),z(5,1)]', (5, 6)),
    ('Vindex(z(2,3,4))[:,0,z(5)]', (5, 2)),
    ('Vindex(z(2,3,4))[:,:,z(5)]', (5, 2, 3)),
    ('Vindex(z(2,3,4))[:,z(5),0]', (5, 2)),
    ('Vindex(z(2,3,4))[:,z(5),:]', (5, 2, 4)),
    ('Vindex(z(2,3,4))[:,z(5),z(5)]', (5, 2)),
    ('Vindex(z(2,3,4))[:,z(5,1),z(6)]', (5, 6, 2)),
    ('Vindex(z(2,3,4))[:,z(6),z(5,1)]', (5, 6, 2)),
    ('Vindex(z(2,3,4))[z(5),0,0]', (5,)),
    ('Vindex(z(2,3,4))[z(5),0,:]', (5, 4)),
    ('Vindex(z(2,3,4))[z(5),:,0]', (5, 3)),
    ('Vindex(z(2,3,4))[z(5),:,:]', (5, 3, 4)),
    ('Vindex(z(2,3,4))[z(5),0,z(5)]', (5,)),
    ('Vindex(z(2,3,4))[z(5,1),0,z(6)]', (5, 6)),
    ('Vindex(z(2,3,4))[z(6),0,z(5,1)]', (5, 6)),
    ('Vindex(z(2,3,4))[z(5),:,z(5)]', (5, 3)),
    ('Vindex(z(2,3,4))[z(5,1),:,z(6)]', (5, 6, 3)),
    ('Vindex(z(2,3,4))[z(6),:,z(5,1)]', (5, 6, 3)),
]


@pytest.mark.parametrize('expression,expected_shape', SHAPE_EXAMPLES, ids=str)
def test_shape(expression, expected_shape):
    result = eval(expression)
    assert result.shape == expected_shape


@pytest.mark.parametrize('event_shape', [(), (7,)], ids=str)
@pytest.mark.parametrize('j_shape', [(), (2,), (3, 1), (4, 1, 1), (4, 3, 2)], ids=str)
@pytest.mark.parametrize('i_shape', [(), (2,), (3, 1), (4, 1, 1), (4, 3, 2)], ids=str)
@pytest.mark.parametrize('x_shape', [(), (2,), (3, 1), (4, 1, 1), (4, 3, 2)], ids=str)
def test_value(x_shape, i_shape, j_shape, event_shape):
    x = torch.rand(x_shape + (5, 6) + event_shape)
    i = dist.Categorical(torch.ones(5)).sample(i_shape)
    j = dist.Categorical(torch.ones(6)).sample(j_shape)
    if event_shape:
        actual = Vindex(x)[..., i, j, :]
    else:
        actual = Vindex(x)[..., i, j]

    shape = broadcast_shape(x_shape, i_shape, j_shape)
    x = x.expand(shape + (5, 6) + event_shape)
    i = i.expand(shape)
    j = j.expand(shape)
    expected = x.new_empty(shape + event_shape)
    for ind in (itertools.product(*map(range, shape)) if shape else [()]):
        expected[ind] = x[ind + (i[ind].item(), j[ind].item())]
    assert_equal(actual, expected)


@pytest.mark.parametrize('prev_enum_dim,curr_enum_dim', [(-3, -4), (-4, -5), (-5, -3)])
def test_hmm_example(prev_enum_dim, curr_enum_dim):
    hidden_dim = 8
    probs_x = torch.rand(hidden_dim, hidden_dim, hidden_dim)
    x_prev = torch.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - prev_enum_dim))
    x_curr = torch.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - curr_enum_dim))

    expected = probs_x[x_prev.unsqueeze(-1), x_curr.unsqueeze(-1), torch.arange(hidden_dim)]
    actual = Vindex(probs_x)[x_prev, x_curr, :]
    assert_equal(actual, expected)


@pytest.mark.parametrize("args,expected", [
    (0, 0),
    (1, 1),
    (None, None),
    (slice(1, 2, 3), slice(1, 2, 3)),
    (Ellipsis, Ellipsis),
    ((0, 1, None, slice(1, 2, 3), Ellipsis), (0, 1, None, slice(1, 2, 3), Ellipsis)),
    (((0, 1), (None, slice(1, 2, 3)), Ellipsis), (0, 1, None, slice(1, 2, 3), Ellipsis)),
    ((Ellipsis, None), (Ellipsis, None)),
    ((Ellipsis, (Ellipsis, None)), (Ellipsis, None)),
    ((Ellipsis, (Ellipsis, None, None)), (Ellipsis, None, None)),
])
def test_index(args, expected):
    assert Index(tensor_mock)[args] == expected
