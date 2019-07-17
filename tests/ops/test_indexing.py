from __future__ import absolute_import, division, print_function

import itertools

import pytest
import torch

import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro.ops.indexing import Vindex
from tests.common import assert_equal


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


@pytest.mark.parametrize('optimize_prev', [True, False])
@pytest.mark.parametrize('optimize_curr', [True, False])
@pytest.mark.parametrize('prev_enum_dim,curr_enum_dim', [(-3, -4), (-4, -5), (-5, -3)])
def test_hmm_example(prev_enum_dim, curr_enum_dim, optimize_prev, optimize_curr):
    hidden_dim = 8
    probs_x = torch.rand(hidden_dim, hidden_dim, hidden_dim)
    x_prev = torch.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - prev_enum_dim))
    x_curr = torch.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - curr_enum_dim))
    if optimize_prev:
        x_prev._pyro_generalized_slice = slice(hidden_dim)
    if optimize_curr:
        x_curr._pyro_generalized_slice = slice(hidden_dim)

    expected = probs_x[x_prev.unsqueeze(-1), x_curr.unsqueeze(-1), torch.arange(hidden_dim)]
    actual = Vindex(probs_x)[x_prev, x_curr, :]
    assert_equal(actual, expected)


def _generalized_arange(size, target_dim=-1):
    assert target_dim < 0
    result = torch.arange(size)
    result = result.reshape((-1,) + (1,) * (1 - target_dim))
    result._pyro_generalized_slice = slice(size)
    return result


def _clone(x):
    return x.clone() if isinstance(x, torch.Tensor) else x


@pytest.mark.parametrize('i', [0, torch.tensor(1), slice(None), _generalized_arange(3, -1)])
@pytest.mark.parametrize('j', [1, torch.tensor(2), slice(None), _generalized_arange(4, -3)])
@pytest.mark.parametrize('k', [2, torch.tensor(0), slice(None), _generalized_arange(5, -2)])
def test_generalized_slice(i, j, k):
    tensor = torch.randn(3, 4, 5)
    actual = Vindex(tensor)[i, j, k]
    assert actual.storage().data_ptr() == tensor.storage().data_ptr()

    # Forget ._pyro_generalized_slice attributes and re-execute.
    i = _clone(i)
    j = _clone(j)
    k = _clone(k)
    expected = Vindex(tensor)[i, j, k]
    assert_equal(actual, expected)
