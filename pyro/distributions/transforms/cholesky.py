# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform

from pyro.distributions.constraints import corr_cholesky_constraint


def _vector_to_l_cholesky(z):
    D = (1.0 + math.sqrt(1.0 + 8.0 * z.shape[-1])) / 2.0
    if D % 1 != 0:
        raise ValueError("Correlation matrix transformation requires d choose 2 inputs")
    D = int(D)
    x = torch.zeros(z.shape[:-1] + (D, D), dtype=z.dtype, device=z.device)

    x[..., 0, 0] = 1
    x[..., 1:, 0] = z[..., :(D - 1)]
    i = D - 1
    last_squared_x = torch.zeros(z.shape[:-1] + (D,), dtype=z.dtype, device=z.device)
    for j in range(1, D):
        distance_to_copy = D - 1 - j
        last_squared_x = last_squared_x[..., 1:] + x[..., j:, (j - 1)].clone()**2
        x[..., j, j] = (1 - last_squared_x[..., 0]).sqrt()
        x[..., (j + 1):, j] = z[..., i:(i + distance_to_copy)] * (1 - last_squared_x[..., 1:]).sqrt()
        i += distance_to_copy
    return x


class CorrLCholeskyTransform(Transform):
    """
    Transforms a vector into the cholesky factor of a correlation matrix.

    The input should have shape `[batch_shape] + [d * (d-1)/2]`. The output will
    have shape `[batch_shape] + [d, d]`.

    References:

    [1] Cholesky Factors of Correlation Matrices. Stan Reference Manual v2.18,
    Section 10.12.

    """
    domain = constraints.real_vector
    codomain = corr_cholesky_constraint
    bijective = True
    sign = +1
    event_dim = 1

    def __eq__(self, other):
        return isinstance(other, CorrLCholeskyTransform)

    def _call(self, x):
        z = x.tanh()
        return _vector_to_l_cholesky(z)

    def _inverse(self, y):
        if (y.shape[-2] != y.shape[-1]):
            raise ValueError("A matrix that isn't square can't be a Cholesky factor of a correlation matrix")
        D = y.shape[-1]

        z_tri = torch.zeros(y.shape[:-2] + (D - 2, D - 2), dtype=y.dtype, device=y.device)
        z_stack = [
            y[..., 1:, 0]
        ]

        for i in range(2, D):
            z_tri[..., i - 2, 0:(i - 1)] = y[..., i, 1:i] / (1 - y[..., i, 0:(i - 1)].pow(2).cumsum(-1)).sqrt()
        for j in range(D - 2):
            z_stack.append(z_tri[..., j:, j])

        z = torch.cat(z_stack, -1)
        return torch.log1p((2 * z) / (1 - z)) / 2

    def log_abs_det_jacobian(self, x, y):
        # Note dependence on pytorch 1.0.1 for batched tril
        tanpart = x.cosh().log().sum(-1).mul(-2)
        matpart = (1 - y.pow(2).cumsum(-1).tril(diagonal=-2)).log().div(2).sum(-1).sum(-1)
        return tanpart + matpart
