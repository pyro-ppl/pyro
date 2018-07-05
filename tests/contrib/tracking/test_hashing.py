from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.contrib.tracking.hashing import LSH, ApproxSet
from tests.common import assert_equal


@pytest.mark.parametrize('scale', [-1., 0., -1 * torch.ones(2, 2)])
def test_lsh_init(scale):
    with pytest.raises(ValueError):
        LSH(scale)


@pytest.mark.parametrize('scale', [0.1, 1, 10, 100])
def test_lsh_add(scale):
    lsh = LSH(scale)
    a = torch.rand(10)
    lsh.add('a', a)
    assert lsh._hash_to_key[lsh._key_to_hash['a']] == {'a'}


@pytest.mark.parametrize('scale', [0.1, 1, 10, 100])
def test_lsh_hash_nearby(scale):
    k = 5
    lsh = LSH(scale)
    a = -2 * scale + torch.rand(k) * scale * 0.49
    b = -1 * scale + torch.rand(k) * scale * 0.49
    c = torch.rand(k) * scale * 0.49
    d = scale + torch.rand(k) * scale * 0.49
    e = 2 * scale + torch.rand(k) * scale * 0.49
    f = 4 * scale + torch.rand(k) * scale * 0.49

    assert_equal(lsh._hash(a), (-2,) * k)
    assert_equal(lsh._hash(b), (-1,) * k)
    assert_equal(lsh._hash(c), (0,) * k)
    assert_equal(lsh._hash(d), (1,) * k)
    assert_equal(lsh._hash(e), (2,) * k)
    assert_equal(lsh._hash(f), (4,) * k)

    lsh.add('a', a)
    lsh.add('b', b)
    lsh.add('c', c)
    lsh.add('d', d)
    lsh.add('e', e)
    lsh.add('f', f)

    assert lsh.nearby('a') == {'b'}
    assert lsh.nearby('b') == {'a', 'c'}
    assert lsh.nearby('c') == {'b', 'd'}
    assert lsh.nearby('d') == {'c', 'e'}
    assert lsh.nearby('e') == {'d'}
    assert lsh.nearby('f') == set()


def test_lsh_overwrite():
    lsh = LSH(1)
    a = torch.zeros(2)
    b = torch.ones(2)
    lsh.add('a', a)
    lsh.add('b', b)
    assert lsh.nearby('a') == {'b'}
    b = torch.ones(2) * 4
    lsh.add('b', b)
    assert lsh.nearby('a') == set()


def test_lsh_remove():
    lsh = LSH(1)
    a = torch.zeros(2)
    b = torch.ones(2)
    lsh.add('a', a)
    lsh.add('b', b)
    assert lsh.nearby('a') == {'b'}
    lsh.remove('b')
    assert lsh.nearby('a') == set()


@pytest.mark.parametrize('scale', [-1., 0., -1 * torch.ones(2, 2)])
def test_aps_init(scale):
    with pytest.raises(ValueError):
        ApproxSet(scale)


@pytest.mark.parametrize('scale', [0.1, 1, 10, 100])
def test_aps_hash(scale):
    k = 10
    aps = ApproxSet(scale)
    a = -2 * scale + torch.rand(k) * scale * 0.49
    b = -1 * scale + torch.rand(k) * scale * 0.49
    c = torch.rand(k) * scale * 0.49
    d = scale + torch.rand(k) * scale * 0.49
    e = 2 * scale + torch.rand(k) * scale * 0.49
    f = 4 * scale + torch.rand(k) * scale * 0.49

    assert_equal(aps._hash(a), (-2,) * k)
    assert_equal(aps._hash(b), (-1,) * k)
    assert_equal(aps._hash(c), (0,) * k)
    assert_equal(aps._hash(d), (1,) * k)
    assert_equal(aps._hash(e), (2,) * k)
    assert_equal(aps._hash(f), (4,) * k)


@pytest.mark.parametrize('scale', [0.1, 1, 10, 100])
def test_aps_try_add(scale):
    k = 10
    aps = ApproxSet(scale)
    a = torch.rand(k) * scale * 0.49
    b = torch.rand(k) * scale * 0.49
    c = scale + torch.rand(k) * scale * 0.49
    d = scale + torch.rand(k) * scale * 0.49

    assert_equal(aps.try_add(a), True)
    assert_equal(aps.try_add(b), False)
    assert_equal(aps.try_add(c), True)
    assert_equal(aps.try_add(d), False)
