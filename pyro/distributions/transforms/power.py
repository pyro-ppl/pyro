# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.transforms import Transform


class PositivePowerTransform(Transform):
    r"""
    Transform via the mapping
    :math:`y=\operatorname{sign}(x)|x|^{\text{exponent}}`.

    Whereas :class:`~torch.distributions.transforms.PowerTransform` allows
    arbitrary ``exponent`` and restricts domain and codomain to postive values,
    this class restricts ``exponent > 0`` and allows real domain and codomain.

    .. warning:: The Jacobian is typically zero or infinite at the origin.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, exponent, *, cache_size=0, validate_args=None):
        super().__init__(cache_size=cache_size)
        if isinstance(exponent, int):
            exponent = float(exponent)
        exponent = torch.as_tensor(exponent)
        self.exponent = exponent
        if validate_args is None:
            validate_args = Distribution._validate_args
        if validate_args:
            if not exponent.gt(0).all():
                raise ValueError(f"Expected exponent > 0 but got:{exponent}")

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return PositivePowerTransform(self.exponent, cache_size=cache_size)

    def __eq__(self, other):
        if not isinstance(other, PositivePowerTransform):
            return False
        return self.exponent.eq(other.exponent).all().item()

    def _call(self, x):
        return x.abs().pow(self.exponent) * x.sign()

    def _inverse(self, y):
        return y.abs().pow(self.exponent.reciprocal()) * y.sign()

    def log_abs_det_jacobian(self, x, y):
        return self.exponent.log() + (y / x).log()

    def forward_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))

    def inverse_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))
