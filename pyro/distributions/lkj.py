# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch

from pyro.distributions.torch import LKJCholesky, TransformedDistribution
from pyro.distributions.transforms.cholesky import CorrMatrixCholeskyTransform

from . import constraints


class LKJCorrCholesky(LKJCholesky):  # DEPRECATED
    def __init__(self, d, eta, validate_args=None):
        warnings.warn(
            "class LKJCorrCholesky(d, eta, validate_args=None) is deprecated "
            "in favor of LKJCholesky(dim, concentration, validate_args=None).",
            FutureWarning,
        )
        super().__init__(d, concentration=eta, validate_args=validate_args)


class LKJ(TransformedDistribution):
    r"""
    LKJ distribution for correlation matrices. The distribution is controlled by ``concentration``
    parameter :math:`\eta` to make the probability of the correlation matrix :math:`M` propotional
    to :math:`\det(M)^{\eta - 1}`. Because of that, when ``concentration == 1``, we have a
    uniform distribution over correlation matrices.

    When ``concentration > 1``, the distribution favors samples with large large determinent. This
    is useful when we know a priori that the underlying variables are not correlated.
    When ``concentration < 1``, the distribution favors samples with small determinent. This is
    useful when we know a priori that some underlying variables are correlated.

    :param int dimension: dimension of the matrices
    :param ndarray concentration: concentration/shape parameter of the
        distribution (often referred to as eta)

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe
    """

    arg_constraints = {"concentration": constraints.positive}
    support = constraints.corr_matrix

    def __init__(self, dim, concentration=1.0, validate_args=None):
        base_dist = LKJCholesky(dim, concentration)
        self.dim, self.concentration = base_dist.dim, base_dist.concentration
        super(LKJ, self).__init__(
            base_dist, CorrMatrixCholeskyTransform().inv, validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LKJCholesky, _instance)
        return super(LKJCholesky, self).expand(batch_shape, _instance=new)

    @property
    def mean(self):
        return torch.eye(self.dim).expand(self.batch_shape + (self.dim, self.dim))
