# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions import constraints
from torch.distributions.transforms import Transform

from pyro.ops.tensor_utils import dct, idct


class DiscreteCosineTransform(Transform):
    """
    Discrete Cosine Transform of type-II.

    This uses :func:`~pyro.ops.tensor_utils.dct` and
    :func:`~pyro.ops.tensor_utils.idct` to compute
    orthonormal DCT and inverse DCT transforms. The jacobian is 1.

    :param int dim: Dimension along which to transform. Must be negative.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, dim=-1, cache_size=0):
        assert isinstance(dim, int) and dim < 0
        self.event_dim = -dim
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return type(self) == type(other) and self.event_dim == other.event_dim

    def _call(self, x):
        dim = -self.event_dim
        if dim != -1:
            x = x.transpose(dim, -1)
        y = dct(x)
        if dim != -1:
            y = y.transpose(dim, -1)
        return y

    def _inverse(self, y):
        dim = -self.event_dim
        if dim != -1:
            y = y.transpose(dim, -1)
        x = idct(y)
        if dim != -1:
            x = x.transpose(dim, -1)
        return x

    def log_abs_det_jacobian(self, x, y):
        return x.new_zeros((1,) * self.event_dim)
