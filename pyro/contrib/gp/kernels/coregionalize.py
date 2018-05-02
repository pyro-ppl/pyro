from __future__ import absolute_import, division, print_function

import torch
from torch.nn import Parameter

from .kernel import Kernel


class Coregionalize(Kernel):
    r"""
    A kernel for the linear model of coregionalization
    :math:`k(x,z) = x R R^T z^T` where :math:`R` is an
    ``input_dim``-by-``rank`` regionalization matrix and typically
    ``rank < input_dim``.

    This generalizes the
    :class:`~pyro.contrib.gp.kernels.dot_product.Linear` kernel to multiple
    features.

    References:

    [1] Mauricio A. Alvarez, Lorenzo Rosasco, Neil D. Lawrence (2012)
        Kernels for Vector-Valued Functions: a Review

    :param int input_dim: Number of feature dimensions of inputs.
    :param torch.Tensor regions: An optional ``(input_dim, rank)`` shaped
        matrix that maps features to ``rank``-many regions. If unspecified,
        this will be randomly initialized.
    :param int rank: Optional rank. This is only used if ``regions`` is
        unspecified. If unspecified, ``rank`` defaults to ``input_dim``.
    :param list active_dims: List of feature dimensions of the input which the
        kernel acts on.
    :param str name: Name of the kernel.
    """

    def __init__(self, input_dim, regions=None, rank=None, active_dims=None, name="coregionalize"):
        super(Coregionalize, self).__init__(input_dim, active_dims, name)
        if regions is None:
            if rank is None:
                rank = input_dim
            regions = torch.randn(input_dim, rank) / rank ** 0.5
        if regions.dim() != 2 or regions.shape[0] != input_dim:
            raise ValueError("Expected region.shape == ({},r), actual {}".format(regions.shape))
        self.regions = Parameter(regions)

    def forward(self, X, Z=None, diag=False):
        regions = self.get_param("regions")
        X = self._slice_input(X)
        Xc = X.matmul(regions)
        if diag:
            return (Xc ** 2).sum(-1)
        elif Z is None:
            Zc = Xc
        else:
            Z = self._slice_input(Z)
            Zc = Z.matmul(regions)
        return Xc.matmul(Zc.t())
