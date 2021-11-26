# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform


class _UnitLowerCholesky(torch.distributions.Constraint):
    """
    Constrain to lower-triangular square matrices with all ones diagonals.
    """
    event_dim = 2

    def check(self, value):
        value_tril = value.tril()
        lower_triangular = (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]

        ones_diagonal = (value.diagonal(dim1=-2, dim2=-1) == 1).min(-1)[0]
        return lower_triangular & ones_diagonal


class UnitLowerCholeskyTransform(Transform):
    """
    Transform from unconstrained matrices to lower-triangular matrices with
    all ones diagonals.
    """
    domain = constraints.independent(constraints.real, 2)
    codomain = constraints.unit_lower_cholesky

    def __eq__(self, other):
        return isinstance(other, UnitLowerCholeskyTransform)

    def _call(self, x):
        return x.tril(-1) + torch.eye(x.size(-1), device=x.device, dtype=x.dtype)

    def _inverse(self, y):
        return y.tril(-1)


__all__ = [
    "_UnitLowerCholesky",
    "UnitLowerCholeskyTransform",
]
