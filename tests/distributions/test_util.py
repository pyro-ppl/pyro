from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import torch

from pyro.distributions.util import broadcast_shape, sum_leftmost, sum_rightmost

INF = float('inf')


@pytest.mark.parametrize('shapes', [
    ([],),
    ([1],),
    ([2],),
    ([], []),
    ([], [1]),
    ([], [2]),
    ([1], []),
    ([2], []),
    ([1], [2]),
    ([2], [1]),
    ([2], [2]),
    ([2], [3, 1]),
    ([2, 1], [3]),
    ([2, 1], [1, 3]),
    ([1, 2, 4, 1, 3], [6, 7, 1, 1, 5, 1]),
    ([], [3, 1], [2], [4, 3, 1], [5, 4, 1, 1]),
])
def test_broadcast_shape(shapes):
    assert broadcast_shape(*shapes) == np.broadcast(*map(np.empty, shapes)).shape


@pytest.mark.parametrize('shapes', [
    ([3], [4]),
    ([2, 1], [1, 3, 1]),
])
def test_broadcast_shape_error(shapes):
    with pytest.raises((ValueError, RuntimeError)):
        broadcast_shape(*shapes)


@pytest.mark.parametrize('shapes', [
    ([],),
    ([1],),
    ([2],),
    ([], []),
    ([], [1]),
    ([], [2]),
    ([1], []),
    ([2], []),
    ([1], [1]),
    ([2], [2]),
    ([2], [2]),
    ([2], [3, 2]),
    ([2, 3], [3]),
    ([2, 3], [2, 3]),
    ([4], [1, 2, 3, 4], [2, 3, 4], [3, 4]),
])
def test_broadcast_shape_strict(shapes):
    assert broadcast_shape(*shapes, strict=True) == np.broadcast(*map(np.empty, shapes)).shape


@pytest.mark.parametrize('shapes', [
    ([1], [2]),
    ([2], [1]),
    ([3], [4]),
    ([2], [3, 1]),
    ([2, 1], [3]),
    ([2, 1], [1, 3]),
    ([2, 1], [1, 3, 1]),
    ([1, 2, 4, 1, 3], [6, 7, 1, 1, 5, 1]),
    ([], [3, 1], [2], [4, 3, 1], [5, 4, 1, 1]),
])
def test_broadcast_shape_strict_error(shapes):
    with pytest.raises(ValueError):
        broadcast_shape(*shapes, strict=True)


def test_sum_rightmost():
    x = torch.ones(2, 3, 4)
    assert sum_rightmost(x, 0).shape == (2, 3, 4)
    assert sum_rightmost(x, 1).shape == (2, 3)
    assert sum_rightmost(x, 2).shape == (2,)
    assert sum_rightmost(x, -1).shape == (2,)
    assert sum_rightmost(x, -2).shape == (2, 3)
    assert sum_rightmost(x, INF).shape == ()


def test_sum_leftmost():
    x = torch.ones(2, 3, 4)
    assert sum_leftmost(x, 0).shape == (2, 3, 4)
    assert sum_leftmost(x, 1).shape == (3, 4)
    assert sum_leftmost(x, 2).shape == (4,)
    assert sum_leftmost(x, -1).shape == (4,)
    assert sum_leftmost(x, -2).shape == (3, 4)
    assert sum_leftmost(x, INF).shape == ()
