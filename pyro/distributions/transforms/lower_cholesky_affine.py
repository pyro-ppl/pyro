# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform

from pyro.distributions.util import copy_docs_from


@copy_docs_from(Transform)
class LowerCholeskyAffine(Transform):
    """
    A bijection of the form,

        :math:`\\mathbf{y} = \\mathbf{L} \\mathbf{x} + \\mathbf{r}`

    where `\\mathbf{L}` is a lower triangular matrix and `\\mathbf{r}` is a vector.

    :param loc: the fixed D-dimensional vector to shift the input by.
    :type loc: torch.tensor
    :param scale_tril: the D x D lower triangular matrix used in the transformation.
    :type scale_tril: torch.tensor

    """
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1
    volume_preserving = False

    def __init__(self, loc, scale_tril, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale_tril = scale_tril
        assert loc.size(-1) == scale_tril.size(-1) == scale_tril.size(-2), \
            "loc and scale_tril must be of size D and D x D, respectively (instead: {}, {})".format(loc.shape,
                                                                                                    scale_tril.shape)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        return torch.matmul(self.scale_tril, x.unsqueeze(-1)).squeeze(-1) + self.loc

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        return torch.triangular_solve((y - self.loc).unsqueeze(-1), self.scale_tril,
                                      upper=False, transpose=False)[0].squeeze(-1)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs(dy/dx)).
        """
        return torch.ones(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device) * \
            self.scale_tril.diag().log().sum()

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return LowerCholeskyAffine(self.loc, self.scale_tril, cache_size=cache_size)
