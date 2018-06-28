from __future__ import absolute_import, division, print_function

import heapq
import itertools
from collections import defaultdict
from numbers import Number

import torch


class LSH(object):
    """
    Implements locality-sensitive hash for low-dimensional euclidean space.
    """
    def __init__(self, scale):
        assert scale > 0 if isinstance(scale, Number) else (scale > 0).all(), \
            "scale must be greater than 0, given:{}".format(scale)
        self._scale = scale
        self._hash_to_key = defaultdict(set)
        self._key_to_hash = {}

    def _hash(self, point):
        coords = (point.detach().cpu() / self._scale).round()
        return tuple(map(int, coords))

    def add(self, key, point):
        _hash = self._hash(point)
        self._key_to_hash[key] = _hash
        self._hash_to_key[_hash].add(key)

    def remove(self, key):
        _hash = self._key_to_hash.pop(key)
        self._hash_to_key[_hash].remove(key)

    def nearby(self, key):
        _hash = self._key_to_hash[key]
        result = set()
        for nearby_hash in itertools.product(*[[i - 1, i, i + 1] for i in _hash]):
            result |= self._hash_to_key[nearby_hash]
        result.remove(key)
        return result


class ApproxSet(object):
    """
    Queries low-dimensional euclidean space for approximate occupancy.
    """
    def __init__(self, scale):
        self._scale = scale
        self._bins = set()

    def _hash(self, point):
        coords = (point.detach().cpu() / self._scale).round()
        return tuple(map(int, coords))

    def try_add(self, point):
        """
        If bin is unoccupied, adds to bin and returns True.
        If bin is already occupied, returns False.
        """
        _hash = self._hash(point)
        if _hash in self._bins:
            return False
        self._bins.add(_hash)
        return True
