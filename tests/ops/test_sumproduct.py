from __future__ import absolute_import, division, print_function

import pytest

from pyro.ops.sumproduct import zip_align_right


@pytest.mark.parametrize('xs,ys,expected', [
    (['a', 'b'], [6, 5], [('a', 6), ('b', 5)]),
    (['b'], [6, 5], [('b', 5)]),
    (['a', 'b'], [5], [('b', 5)]),
])
def test_zip_align_right(xs, ys, expected):
    actual = list(zip_align_right(xs, ys))
    assert actual == expected
