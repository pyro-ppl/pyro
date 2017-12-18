from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from pyro.distributions.util import broadcast_shape


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
