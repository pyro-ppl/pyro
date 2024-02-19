# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import torch
from torch.distributions.transforms import CorrCholeskyTransform, Transform

from .. import constraints


class CorrLCholeskyTransform(CorrCholeskyTransform):  # DEPRECATED
    def __init__(self, cache_size=0):
        warnings.warn(
            "class CorrLCholeskyTransform is deprecated in favor of CorrCholeskyTransform.",
            FutureWarning,
        )
        super().__init__(cache_size=cache_size)


class CholeskyTransform(Transform):
    r"""
    Transform via the mapping :math:`y = safe_cholesky(x)`, where `x` is a
    positive definite matrix.
    """

    bijective = True
    domain = constraints.positive_definite
    codomain = constraints.lower_cholesky

    def __eq__(self, other):
        return isinstance(other, CholeskyTransform)

    def _call(self, x):
        return torch.linalg.cholesky(x)

    def _inverse(self, y):
        return torch.matmul(y, torch.transpose(y, -2, -1))

    def log_abs_det_jacobian(self, x, y):
        # Ref: http://web.mit.edu/18.325/www/handouts/handout2.pdf page 13
        n = x.shape[-1]
        order = torch.arange(n, 0, -1, dtype=x.dtype, device=x.device)
        return -n * math.log(2) - (
            order * torch.diagonal(y, dim1=-2, dim2=-1).log()
        ).sum(-1)


class CorrMatrixCholeskyTransform(CholeskyTransform):
    r"""
    Transform via the mapping :math:`y = safe_cholesky(x)`, where `x` is a
    correlation matrix.
    """

    bijective = True
    domain = constraints.corr_matrix
    # TODO: change corr_cholesky_constraint to corr_cholesky when the latter is availabler
    codomain = constraints.corr_cholesky_constraint

    def __eq__(self, other):
        return isinstance(other, CorrMatrixCholeskyTransform)

    def log_abs_det_jacobian(self, x, y):
        # NB: see derivation in LKJCholesky implementation
        n = x.shape[-1]
        order = torch.arange(n - 1, -1, -1, dtype=x.dtype, device=x.device)
        return -(order * torch.diagonal(y, dim1=-2, dim2=-1).log()).sum(-1)
