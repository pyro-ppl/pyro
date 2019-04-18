from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.ops.indexing import broadcasted
from tests.common import assert_equal


def z(*args):
    return torch.zeros(*args, dtype=torch.long)


SHAPE_EXAMPLES = [
    ('broadcasted(z(()))[...]', ()),
    ('broadcasted(z(2))[...]', (2,)),
    ('broadcasted(z(2))[:]', (2,)),
    ('broadcasted(z(2))[z(3)]', (3,)),
    ('broadcasted(z(2,3))[...]', (2, 3)),
    ('broadcasted(z(2,3))[...,z(2)]', (2,)),
    ('broadcasted(z(2,3))[...,z(4,1)]', (4, 2)),
    ('broadcasted(z(2,3))[:,z(4)]', (4, 2)),
    ('broadcasted(z(2,3))[z(4),:]', (4, 3)),
    ('broadcasted(z(2,3))[z(4)]', (4, 3)),
    ('broadcasted(z(2,3))[z(4),z(4)]', (4,)),
    ('broadcasted(z(2,3))[z(5,1),z(4)]', (5, 4)),
    ('broadcasted(z(2,3))[z(4),z(5,1)]', (5, 4)),
    ('broadcasted(z(2,3,4))[:,:,:]', (2, 3, 4)),
]


@pytest.mark.parametrize('expression,expected_shape', SHAPE_EXAMPLES)
def test_shape(expression, expected_shape):
    result = eval(expression)
    assert result.shape == expected_shape


@pytest.mark.parametrize('prev_enum_dim,curr_enum_dim', [(-3, -4), (-4, -5), (-5, -3)])
def test_hmm_example(prev_enum_dim, curr_enum_dim):
    hidden_dim = 8
    probs_x = torch.rand(hidden_dim, hidden_dim, hidden_dim)
    x_prev = torch.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - prev_enum_dim))
    x_curr = torch.arange(hidden_dim).reshape((-1,) + (1,) * (-1 - curr_enum_dim))

    expected = probs_x[x_prev.unsqueeze(-1), x_curr.unsqueeze(-1), torch.arange(hidden_dim)]
    actual = broadcasted(probs_x)[x_prev, x_curr, :]
    assert_equal(actual, expected)
