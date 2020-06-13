# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions import constraints
from torch.distributions.transforms import Transform

from pyro.ops.tensor_utils import haar_transform, inverse_haar_transform


class HaarTransform(Transform):
    """
    Discrete Haar transform.

    This uses :func:`~pyro.ops.tensor_utils.haar_transform` and
    :func:`~pyro.ops.tensor_utils.inverse_haar_transform` to compute
    (orthonormal) Haar and inverse Haar transforms. The jacobian is 1.
    For sequences with length `T` not a power of two, this implementation
    is equivalent to a block-structured Haar transform in which block
    sizes decrease by factors of one half from left to right.

    :param int dim: Dimension along which to transform. Must be negative.
        This is an absolute dim counting from the right.
    :param bool flip: Whether to flip the time axis before applying the
        Haar transform. Defaults to false.
    :param experimental_event_dim: EXPERIMENTAL Optional ``event_dim``,
        overriding the default of ``event_dim = -dim``. This results in a
        proper transform only if ``event_dim >= -dim``; however an improper
        transform (that mixes elements across batches) can still be used in
        some applications, such as reparameterization without subsampling.
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, dim=-1, flip=False, *,
                 experimental_event_dim=None, cache_size=0):
        assert isinstance(dim, int) and dim < 0
        self.dim = dim
        self.flip = flip
        self.event_dim = -dim
        if experimental_event_dim is not None:
            assert experimental_event_dim >= 0
            self.event_dim = experimental_event_dim
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return (type(self) == type(other) and self.dim == other.dim and
                self.flip == other.flip and self.event_dim == other.event_dim)

    def _call(self, x):
        dim = self.dim
        if dim != -1:
            x = x.transpose(dim, -1)
        if self.flip:
            x = x.flip(-1)
        y = haar_transform(x)
        if dim != -1:
            y = y.transpose(dim, -1)
        return y

    def _inverse(self, y):
        dim = self.dim
        if dim != -1:
            y = y.transpose(dim, -1)
        x = inverse_haar_transform(y)
        if self.flip:
            x = x.flip(-1)
        if dim != -1:
            x = x.transpose(dim, -1)
        return x

    def log_abs_det_jacobian(self, x, y):
        return x.new_zeros(x.shape[:x.dim() - self.event_dim])

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return HaarTransform(self.dim, flip=self.flip, cache_size=cache_size,
                             experimental_event_dim=self.experimental_event_dim)
