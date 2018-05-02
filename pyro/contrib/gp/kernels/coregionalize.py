from __future__ import absolute_import, division, print_function

import torch
from torch.nn import Parameter

from .kernel import Kernel


class Coregionalize(Kernel):
    r"""
    A kernel for the linear model of coregionalization
    :math:`k(x,z) = x^T R R^T z` where :math:`R` is an
    ``input_dim``-by-``rank`` matrix and typically ``rank < input_dim``.

    This generalizes the
    :class:`~pyro.contrib.gp.kernels.dot_product.Linear` kernel to multiple
    features. The typical use case is for modeling correlations among outputs
    of a multi-output GP, where outputs are coded as distinct data points with
    one-hot coded features denoting which output each datapoint represents.

    References:

    [1] Mauricio A. Alvarez, Lorenzo Rosasco, Neil D. Lawrence (2012)
        Kernels for Vector-Valued Functions: a Review

    :param int input_dim: Number of feature dimensions of inputs.
    :param torch.Tensor components: An optional ``(input_dim, rank)`` shaped
        matrix that maps features to ``rank``-many components. If unspecified,
        this will be randomly initialized.
    :param int rank: Optional rank. This is only used if ``components`` is
        unspecified. If unspecified, ``rank`` defaults to ``input_dim``.
    :param list active_dims: List of feature dimensions of the input which the
        kernel acts on.
    :param str name: Name of the kernel.
    """

    def __init__(self, input_dim, components=None, rank=None, active_dims=None, name="coregionalize"):
        super(Coregionalize, self).__init__(input_dim, active_dims, name)
        if components is None:
            if rank is None:
                rank = input_dim
            components = torch.randn(input_dim, rank) / rank ** 0.5
        if components.dim() != 2 or components.shape[0] != input_dim:
            raise ValueError("Expected region.shape == ({},rank), actual {}".format(input_dim, components.shape))
        self.components = Parameter(components)

    def forward(self, X, Z=None, diag=False):
        components = self.get_param("components")
        X = self._slice_input(X)
        Xc = X.matmul(components)
        if diag:
            return (Xc ** 2).sum(-1)
        elif Z is None:
            Zc = Xc
        else:
            Z = self._slice_input(Z)
            Zc = Z.matmul(components)
        return Xc.matmul(Zc.t())
