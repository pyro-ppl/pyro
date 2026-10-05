# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform

from pyro.distributions.torch import TransformedDistribution, Wishart


class _MatrixInverseTransform(Transform):
    r"""
    Bijective transform mapping a positive-definite matrix to its inverse.

    The log absolute determinant of the Jacobian of the matrix inverse map on
    the space of :math:`p \times p` symmetric matrices is
    :math:`-(p + 1)\log|\det X|`.
    """

    domain = constraints.positive_definite
    codomain = constraints.positive_definite
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, _MatrixInverseTransform)

    def _call(self, x):
        return torch.linalg.inv(x)

    def _inverse(self, y):
        return torch.linalg.inv(y)

    def log_abs_det_jacobian(self, x, y):
        p = x.shape[-1]
        return -(p + 1) * torch.linalg.slogdet(x).logabsdet


class InverseWishart(TransformedDistribution):
    r"""
    Creates an inverse Wishart distribution parameterized by a symmetric
    positive-definite scale matrix :math:`\Psi` and degrees of freedom
    :math:`\nu`.

    If :math:`X \sim \text{Wishart}(\nu, \Psi^{-1})`, then
    :math:`X^{-1} \sim \text{InverseWishart}(\nu, \Psi)`.

    :param df: real-valued degrees of freedom, larger than ``dim - 1`` where
        ``dim`` is the dimension of the square matrix.
    :param scale_matrix: positive-definite scale matrix :math:`\Psi`.
    """

    arg_constraints = {
        "df": constraints.positive,
        "scale_matrix": constraints.positive_definite,
    }
    support = constraints.positive_definite

    def __init__(self, df, scale_matrix, validate_args=None):
        # If X ~ Wishart(df, precision=Psi), then X^{-1} ~ InverseWishart(df, Psi).
        base_dist = Wishart(df, precision_matrix=scale_matrix)
        super().__init__(
            base_dist, _MatrixInverseTransform(), validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseWishart, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def df(self):
        return self.base_dist.df

    @property
    def scale_matrix(self):
        return self.base_dist.precision_matrix

    @property
    def mean(self):
        p = self.event_shape[-1]
        return self.scale_matrix / (self.df.unsqueeze(-1).unsqueeze(-1) - p - 1)

    @property
    def mode(self):
        p = self.event_shape[-1]
        return self.scale_matrix / (self.df.unsqueeze(-1).unsqueeze(-1) + p + 1)
