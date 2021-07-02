# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus


def softplus_inv(y):
    return y + y.neg().expm1().neg().log()


# Backport of https://github.com/pytorch/pytorch/pull/52300
class SoftplusTransform(Transform):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        return softplus(x)

    def _inverse(self, y):
        return softplus_inv(y)

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x)


class SoftplusLowerCholeskyTransform(Transform):
    """
    Transform from unconstrained matrices to lower-triangular matrices with
    nonnegative diagonal entries. This is useful for parameterizing positive
    definite matrices in terms of their Cholesky factorization.
    """

    domain = constraints.independent(constraints.real, 2)
    codomain = constraints.lower_cholesky

    def __eq__(self, other):
        return isinstance(other, SoftplusLowerCholeskyTransform)

    def _call(self, x):
        diag = softplus(x.diagonal(dim1=-2, dim2=-1))
        return x.tril(-1) + diag.diag_embed()

    def _inverse(self, y):
        diag = softplus_inv(y.diagonal(dim1=-2, dim2=-1))
        return y.tril(-1) + diag.diag_embed()


__all__ = [
    "SoftplusTransform",
    "SoftplusLowerCholeskyTransform",
]
