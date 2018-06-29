from __future__ import absolute_import, division, print_function

import itertools
from collections import defaultdict
from numbers import Number


class LSH(object):
    """
    Implements locality-sensitive hashing. Allows to efficiently find neighbours
    of a point. Provides 2 guarantees:
    - Difference between coordinates of points not returned by :meth:`nearby`
    and input point is larger than `radius`.
    - Difference between coordinates of points returned by :meth:`nearby` and
    input point is smaller than 2`radius`.
    :param float radius: scaling parameter used in hash function. Determines
    the size of the neighbourhood.

    Example::
        radius = 1
        lsh = LSH(radius)
        a = torch.tensor([-0.49, -0.49]) # hash(a)=(0,0)
        b = torch.tensor([1.0, 1.0]) # hash(b)=(1,1)
        c = torch.tensor([2.49, 2.49]) # hash(c)=(2,2)
        lsh.add('a', a)
        lsh.add('b', b)
        lsh.add('c', c)
        lsh.nearby('a') # {'b'}
        lsh.nearby('b') # {'a','c'}
        lsh.remove('b')
        lsh.nearby('a') # {}
    """
    def __init__(self, radius):
        assert radius > 0 if isinstance(radius, Number) else (radius > 0).all(), \
            "radius must be greater than 0, given: {}".format(radius)
        self._radius = radius
        self._hash_to_key = defaultdict(set)
        self._key_to_hash = {}

    def _hash(self, point):
        coords = (point.detach().cpu() / self._radius).round()
        return tuple(map(int, coords))

    def add(self, key, point):
        """
        Adds (`key`, `hash(point)`) pair to the hash
        :param key: key used identify point
        :param point: data, should be hashable
        """
        _hash = self._hash(point)
        if key in self._key_to_hash.keys():
            self.remove(key)
        self._key_to_hash[key] = _hash
        self._hash_to_key[_hash].add(key)

    def remove(self, key):
        """
        Removes (`key`, `hash(point)`) pair from the hash.
        Raises KeyError if key is not in hash.
        :param key: key used to identify point
        """
        _hash = self._key_to_hash.pop(key)
        self._hash_to_key[_hash].remove(key)

    def nearby(self, key):
        """
        Returns a set of keys which are neighbours of the input point
        identified by `key`. It returns all points where `abs(point[k]-input[k])<radius`,
        and some points (all points not guaranteed) where `abs(point[k]-input[k])<2*radius`.
        :param key: key used to identify input point
        :return: set of keys identifying neighbours of the input point
        :rtype: set
        """
        _hash = self._key_to_hash[key]
        result = set()
        for nearby_hash in itertools.product(*[[i - 1, i, i + 1] for i in _hash]):
            result |= self._hash_to_key[nearby_hash]
        result.remove(key)
        return result


class ApproxSet(object):
    """
    Queries low-dimensional euclidean space for approximate occupancy.
    :param float radius: scaling parameter used in hash function. Determines
    the size of the neighbourhood.
    """
    def __init__(self, radius):
        assert radius > 0 if isinstance(radius, Number) else (radius > 0).all(), \
            "radius must be greater than 0, given: {}".format(radius)
        self._radius = radius
        self._bins = set()

    def _hash(self, point):
        coords = (point.detach().cpu() / self._radius).round()
        return tuple(map(int, coords))

    def try_add(self, point):
        """
        Attempts to add the point to hash. Only adds if neighboorhood of `point`
        is unoccupied.
        :param torch.Tensor point: point to be queried.
        :return: `True` if point's bin is unoccupied, `False` if bin is already
        occupied.
        :rtype: bool
        """
        _hash = self._hash(point)
        if _hash in self._bins:
            return False
        self._bins.add(_hash)
        return True
