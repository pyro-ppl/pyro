# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numbers

from torch.distributions.transforms import Transform

from pyro.ops.tensor_utils import safe_normalize

from .. import constraints


class Normalize(Transform):
    """
    Safely project a vector onto the sphere wrt the ``p`` norm. This avoids
    the singularity at zero by mapping to the vector ``[1, 0, 0, ..., 0]``.
    """

    domain = constraints.real_vector
    codomain = constraints.sphere
    bijective = False

    def __init__(self, p=2, cache_size=0):
        assert isinstance(p, numbers.Number)
        assert p >= 0
        self.p = p
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return type(self) == type(other) and self.p == other.p

    def _call(self, x):
        return safe_normalize(x, p=self.p)

    def _inverse(self, y):
        return y

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return Normalize(self.p, cache_size=cache_size)
