# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions import constraints
from torch.distributions.transforms import Transform

from pyro.ops.tensor_utils import dct_ii, idct_ii


class DiscreteCosineTransform(Transform):
    """
    Discrete Cosine Transform of type-II.

    This uses :func:`~pyro.ops.tensor_utils.dct_ii` and
    :func:`~pyro.ops.tensor_utils.idct_ii` internally.

    :param int dim: Dimension along which to transform. Must be negative.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1  # TODO verify this

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
        y = dct_ii(x)
        if dim != -1:
            y = y.transpose(dim, -1)
        return y

    def _inverse(self, y):
        dim = -self.event_dim
        if dim != -1:
            y = y.transpose(dim, -1)
        x = idct_ii(y)
        if dim != -1:
            x = x.transpose(dim, -1)
        return x

    def _log_abs_det_jacobian(self, x, y):
        return x.new_zeros((1,) * self.event_dim)
