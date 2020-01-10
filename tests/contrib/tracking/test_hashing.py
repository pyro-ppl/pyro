# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch

from pyro.contrib.tracking.hashing import LSH, ApproxSet, merge_points
from tests.common import assert_equal

logger = logging.getLogger(__name__)


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


def test_merge_points_small():
    points = torch.tensor([
        [0., 0.],
        [0., 1.],
        [2., 0.],
        [2., 0.5],
        [2., 1.0],
    ])
    merged_points, groups = merge_points(points, radius=1.0)

    assert len(merged_points) == 3
    assert set(map(frozenset, groups)) == set(map(frozenset, [[0], [1], [2, 3, 4]]))
    assert_equal(merged_points[0], points[0])
    assert_equal(merged_points[1], points[1])
    assert merged_points[2, 0] == 2
    assert 0.325 <= merged_points[2, 1] <= 0.625


@pytest.mark.parametrize('radius', [0.01, 0.1, 1., 10., 100.])
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_merge_points_large(dim, radius):
    points = 10 * torch.randn(200, dim)
    merged_points, groups = merge_points(points, radius)
    logger.debug('merged {} -> {}'.format(len(points), len(merged_points)))

    assert merged_points.dim() == 2
    assert merged_points.shape[-1] == dim
    assert len(groups) == len(merged_points)
    assert sum(len(g) for g in groups) == len(points)
    assert set(sum(groups, ())) == set(range(len(points)))
    d2 = (merged_points.unsqueeze(-2) - merged_points.unsqueeze(-3)).pow(2).sum(-1)
    assert d2.min() < radius ** 2
