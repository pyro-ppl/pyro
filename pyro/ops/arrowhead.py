# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import torch


SymmArrowhead = namedtuple("SymmArrowhead", ["top", "bottom_diag"])
TriuArrowhead = namedtuple("TriuArrowhead", ["top", "bottom_diag"])


def sqrt(x):
    """
    EXPERIMENTAL Computes the upper triangular square root of an
    symmetric arrowhead matrix.

    :param SymmArrowhead x: an symmetric arrowhead matrix
    :return: the square root of `x`
    :rtype: TriuArrowhead
    """
    assert isinstance(x, SymmArrowhead)
    head_size = x.top.size(0)
    if head_size == 0:
        return TriuArrowhead(x.top, x.bottom_diag.sqrt())

    A, B = x.top[:, :head_size], x.top[:, head_size:]
    # NB: the complexity is O(N * head_size^2)
    # ref: https://en.wikipedia.org/wiki/Schur_complement#Background
    Dsqrt = x.bottom_diag.sqrt()

    # On cholesky error, retry with smaller tail part B.
    num_attempts = 6
    for i in range(num_attempts):
        B_Dsqrt = B / Dsqrt.unsqueeze(-2)  # shape: head_size x N
        schur_complement = A - B_Dsqrt.matmul(B_Dsqrt.t())  # complexity: head_size^2 x N
        # we will decompose schur_complement to U @ U.T (so that the sqrt matrix
        # is upper triangular) using some `flip` operators:
        #   flip(cholesky(flip(schur_complement)))
        try:
            top_left = torch.flip(torch.cholesky(torch.flip(schur_complement, (-2, -1))), (-2, -1))
            break
        except RuntimeError:
            B = B / 2
            continue
        raise RuntimeError("Singular schur complement in computing Cholesky of the input"
                           " arrowhead matrix")

    top_right = B_Dsqrt
    top = torch.cat([top_left, top_right], -1)
    bottom_diag = Dsqrt
    return TriuArrowhead(top, bottom_diag)


def triu_inverse(x):
    """
    EXPERIMENTAL Computes the inverse of an upper-triangular arrowhead matrix.

    :param TriuArrowhead x: an upper-triangular arrowhead matrix.
    :return: the inverse of `x`
    :rtype: TriuArrowhead
    """
    assert isinstance(x, TriuArrowhead)
    head_size = x.top.size(0)
    if head_size == 0:
        return TriuArrowhead(x.top, x.bottom_diag.reciprocal())

    A, B = x.top[:, :head_size], x.top[:, head_size:]
    B_Dinv = B / x.bottom_diag.unsqueeze(-2)

    identity = torch.eye(head_size, dtype=A.dtype, device=A.device)
    top_left = torch.triangular_solve(identity, A, upper=True)[0]
    top_right = -top_left.matmul(B_Dinv)  # complexity: head_size^2 x N
    top = torch.cat([top_left, top_right], -1)
    bottom_diag = x.bottom_diag.reciprocal()
    return TriuArrowhead(top, bottom_diag)


def triu_matvecmul(x, y, transpose=False):
    """
    EXPERIMENTAL Computes matrix-vector product of an upper-triangular
    arrowhead matrix `x` and a vector `y`.

    :param TriuArrowhead x: an upper-triangular arrowhead matrix.
    :param torch.Tensor y: a 1D tensor
    :return: matrix-vector product of `x` and `y`
    :rtype: TriuArrowhead
    """
    assert isinstance(x, TriuArrowhead)
    head_size = x.top.size(0)
    if transpose:
        z = x.top.transpose(-2, -1).matmul(y[:head_size])
        # here we exploit the diagonal structure of the bottom right part
        # of arrowhead_sqrt matrix; so the complexity is still O(N)
        top = z[:head_size]
        bottom = z[head_size:] + x.bottom_diag * y[head_size:]
    else:
        top = x.top.matmul(y)
        bottom = x.bottom_diag * y[head_size:]
    return torch.cat([top, bottom], 0)


def triu_gram(x):
    """
    EXPERIMENTAL Computes the gram matrix `x.T @ x` from an upper-triangular
    arrowhead matrix `x`.

    :param TriuArrowhead x: an upper-triangular arrowhead matrix.
    :return: the square of `x`
    :rtype: TriuArrowhead
    """
    assert isinstance(x, TriuArrowhead)
    head_size = x.top.size(0)
    if head_size == 0:
        return x.bottom_diag.pow(2)

    A, B = x.top[:, :head_size], x.top[:, head_size:]
    top = A.t().matmul(x.top)
    bottom_left = top[:, head_size:].t()
    # the following matmul operator is O(N^2 x head_size)
    bottom_right = B.t().matmul(B) + x.bottom_diag.pow(2).diag()
    return torch.cat([top, torch.cat([bottom_left, bottom_right], -1)], 0)
