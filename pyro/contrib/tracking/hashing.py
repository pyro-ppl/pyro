# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import heapq
import itertools
from collections import defaultdict
from numbers import Number

import torch


class LSH:
    """
    Implements locality-sensitive hashing for low-dimensional euclidean space.


    Allows to efficiently find neighbours of a point. Provides 2 guarantees:

    -   Difference between coordinates of points not returned by :meth:`nearby`
        and input point is larger than ``radius``.
    -   Difference between coordinates of points returned by :meth:`nearby` and
        input point is smaller than 2 ``radius``.

    Example:

        >>> radius = 1
        >>> lsh = LSH(radius)
        >>> a = torch.tensor([-0.51, -0.51]) # hash(a)=(-1,-1)
        >>> b = torch.tensor([-0.49, -0.49]) # hash(a)=(0,0)
        >>> c = torch.tensor([1.0, 1.0]) # hash(b)=(1,1)
        >>> lsh.add('a', a)
        >>> lsh.add('b', b)
        >>> lsh.add('c', c)
        >>> # even though c is within 2radius of a
        >>> lsh.nearby('a') # doctest: +SKIP
        {'b'}
        >>> lsh.nearby('b') # doctest: +SKIP
        {'a', 'c'}
        >>> lsh.remove('b')
        >>> lsh.nearby('a') # doctest: +SKIP
        set()


    :param float radius: Scaling parameter used in hash function. Determines the size of the neighbourhood.

    """
    def __init__(self, radius):
        if not (isinstance(radius, Number) and radius > 0):
            raise ValueError("radius must be float greater than 0, given: {}".format(radius))
        self._radius = radius
        self._hash_to_key = defaultdict(set)
        self._key_to_hash = {}

    def _hash(self, point):
        coords = (point / self._radius).round()
        return tuple(map(int, coords))

    def add(self, key, point):
        """
        Adds (``key``, ``point``) pair to the hash.


        :param key: Key used identify ``point``.
        :param torch.Tensor point: data, should be detached and on cpu.
        """
        _hash = self._hash(point)
        if key in self._key_to_hash:
            self.remove(key)
        self._key_to_hash[key] = _hash
        self._hash_to_key[_hash].add(key)

    def remove(self, key):
        """
        Removes ``key`` and corresponding point from the hash.


        Raises :exc:`KeyError` if key is not in hash.


        :param key: key used to identify point.
        """
        _hash = self._key_to_hash.pop(key)
        self._hash_to_key[_hash].remove(key)

    def nearby(self, key):
        r"""
        Returns a set of keys which are neighbours of the point identified by ``key``.


        Two points are nearby if difference of each element of their hashes is smaller than 2. In euclidean space, this
        corresponds to all points :math:`\mathbf{p}` where :math:`|\mathbf{p}_k-(\mathbf{p_{key}})_k|<r`,
        and some points (all points not guaranteed) where :math:`|\mathbf{p}_k-(\mathbf{p_{key}})_k|<2r`.


        :param key: key used to identify input point.
        :return: a set of keys identifying neighbours of the input point.
        :rtype: set
        """
        _hash = self._key_to_hash[key]
        result = set()
        for nearby_hash in itertools.product(*[[i - 1, i, i + 1] for i in _hash]):
            result |= self._hash_to_key[nearby_hash]
        result.remove(key)
        return result


class ApproxSet:
    """
    Queries low-dimensional euclidean space for approximate occupancy.


    :param float radius: scaling parameter used in hash function. Determines the size of the bin.
                         See :class:`LSH` for details.
    """
    def __init__(self, radius):
        if not (isinstance(radius, Number) and radius > 0):
            raise ValueError("radius must be float greater than 0, given: {}".format(radius))
        self._radius = radius
        self._bins = set()

    def _hash(self, point):
        coords = (point / self._radius).round()
        return tuple(map(int, coords))

    def try_add(self, point):
        """
        Attempts to add ``point`` to set. Only adds there are no points in the ``point``'s bin.


        :param torch.Tensor point: Point to be queried, should be detached and on cpu.
        :return: ``True`` if point is successfully added, ``False`` if there is already a point in ``point``'s bin.
        :rtype: bool
        """
        _hash = self._hash(point)
        if _hash in self._bins:
            return False
        self._bins.add(_hash)
        return True


def merge_points(points, radius):
    """
    Greedily merge points that are closer than given radius.

    This uses :class:`LSH` to achieve complexity that is linear in the number
    of merged clusters and quadratic in the size of the largest merged cluster.

    :param torch.Tensor points: A tensor of shape ``(K,D)`` where ``K`` is
        the number of points and ``D`` is the number of dimensions.
    :param float radius: The minimum distance nearer than which
        points will be merged.
    :return: A tuple ``(merged_points, groups)`` where ``merged_points`` is a
        tensor of shape ``(J,D)`` where ``J <= K``, and ``groups`` is a list of
        tuples of indices mapping merged points to original points. Note that
        ``len(groups) == J`` and ``sum(len(group) for group in groups) == K``.
    :rtype: tuple
    """
    if points.dim() != 2:
        raise ValueError('Expected points.shape == (K,D), but got {}'.format(points.shape))
    if not (isinstance(radius, Number) and radius > 0):
        raise ValueError('Expected radius to be a positive number, but got {}'.format(radius))
    radius = 0.99 * radius  # avoid merging points exactly radius apart, e.g. grid points
    threshold = radius ** 2

    # setup data structures to cheaply search for nearest pairs
    lsh = LSH(radius)
    priority_queue = []
    groups = [(i,) for i in range(len(points))]
    for i, point in enumerate(points):
        lsh.add(i, point)
        for j in lsh.nearby(i):
            d2 = (point - points[j]).pow(2).sum().item()
            if d2 < threshold:
                heapq.heappush(priority_queue, (d2, j, i))
    if not priority_queue:
        return points, groups

    # convert from dense to sparse representation
    next_id = len(points)
    points = dict(enumerate(points))
    groups = dict(enumerate(groups))

    # greedily merge
    while priority_queue:
        d1, i, j = heapq.heappop(priority_queue)
        if i not in points or j not in points:
            continue
        k = next_id
        next_id += 1
        points[k] = (points.pop(i) + points.pop(j)) / 2
        groups[k] = groups.pop(i) + groups.pop(j)
        lsh.remove(i)
        lsh.remove(j)
        lsh.add(k, points[k])
        for i in lsh.nearby(k):
            if i == k:
                continue
            d2 = (points[i] - points[k]).pow(2).sum().item()
            if d2 < threshold:
                heapq.heappush(priority_queue, (d2, i, k))

    # convert from sparse to dense representation
    ids = sorted(points.keys())
    points = torch.stack([points[i] for i in ids])
    groups = [groups[i] for i in ids]

    return points, groups
