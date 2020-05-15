# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform

from pyro.ops.tensor_utils import dct, idct

import numpy as np
import math


class DiscreteCosineTransform(Transform):
    """
    Discrete Cosine Transform of type-II.

    This uses :func:`~pyro.ops.tensor_utils.dct` and
    :func:`~pyro.ops.tensor_utils.idct` to compute
    orthonormal DCT and inverse DCT transforms. The jacobian is 1.

    :param int dim: Dimension along which to transform. Must be negative.
        This is an absolute dim counting from the right.
    :param float smooth: Smoothing parameter. When 0, this transforms white
        noise to white noise; when 1 this transforms Brownian noise to to white
        noise; when -1 this transforms violet noise to white noise; etc. Any
        real number is allowed. https://en.wikipedia.org/wiki/Colors_of_noise.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, dim=-1, smooth=0., cache_size=0):
        assert isinstance(dim, int) and dim < 0
        self.event_dim = -dim
        self.smooth = float(smooth)
        self._weight_cache = None
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return (type(self) == type(other) and self.event_dim == other.event_dim
                and self.smooth == other.smooth)

    @torch.no_grad()
    def _weight(self, y):
        size = y.size(-1)
        if self._weight_cache is None or self._weight_cache.size(-1) != size:
            # Weight by frequency**smooth, where the DCT-II frequencies are:
            freq = torch.linspace(0.5, size - 0.5, size, dtype=y.dtype, device=y.device)
            w = freq.pow_(self.smooth)
            w /= w.log().mean().exp()  # Ensure |jacobian| = 1.
            self._weight_cache = w
        return self._weight_cache

    def _call(self, x):
        dim = -self.event_dim
        if dim != -1:
            x = x.transpose(dim, -1)
        y = dct(x)
        if self.smooth:
            y = y * self._weight(y)
        if dim != -1:
            y = y.transpose(dim, -1)
        return y

    def _inverse(self, y):
        dim = -self.event_dim
        if dim != -1:
            y = y.transpose(dim, -1)
        if self.smooth:
            y = y / self._weight(y)
        x = idct(y)
        if dim != -1:
            x = x.transpose(dim, -1)
        return x

    def log_abs_det_jacobian(self, x, y):
        return x.new_zeros((1,) * self.event_dim)


def haar_matrix(n):
    assert 2 ** int(math.log(n, 2.0)) == n
    return _haar_matrix(n) / math.sqrt(n)


def _haar_matrix(n):
    if n > 2:
        h = _haar_matrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    h_n = np.kron(h, [1, 1])
    h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    h = np.vstack((h_n, h_i))
    return h


class HaarTransform(Transform):
    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, dim=-1, size=0., cache_size=0):
        assert isinstance(dim, int) and dim < 0
        self.event_dim = -dim
        self.size = size
        self.mat = torch.from_numpy(haar_matrix(size))
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return (type(self) == type(other) and self.event_dim == other.event_dim)

    def _call(self, x):
        if self.mat.type != x.type:
            self.mat = self.mat.type_as(x)
        dim = -self.event_dim
        if dim != -1:
            x = x.transpose(dim, -1)
        y = torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)
        if dim != -1:
            y = y.transpose(dim, -1)
        return y

    def _inverse(self, y):
        if self.mat.type != y.type:
            self.mat = self.mat.type_as(y)
        dim = -self.event_dim
        if dim != -1:
            y = y.transpose(dim, -1)
        x = torch.matmul(self.mat.t(), y.unsqueeze(-1)).squeeze(-1)
        if dim != -1:
            x = x.transpose(dim, -1)
        return x

    def log_abs_det_jacobian(self, x, y):
        return x.new_zeros((1,) * self.event_dim)
