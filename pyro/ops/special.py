# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import operator

import torch


class _SafeLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.log()

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad / x.clamp(min=torch.finfo(x.dtype).eps)


def safe_log(x):
    """
    Like :func:`torch.log` but avoids infinite gradients at log(0)
    by clamping them to at most ``1 / finfo.eps``.
    """
    return _SafeLog.apply(x)


def log_beta(x, y, tol=0.):
    """
    Computes log Beta function.

    When ``tol >= 0.02`` this uses a shifted Stirling's approximation to the
    log Beta function. The approximation adapts Stirling's approximation of the
    log Gamma function::

        lgamma(z) ≈ (z - 1/2) * log(z) - z + log(2 * pi) / 2

    to approximate the log Beta function::

        log_beta(x, y) ≈ ((x-1/2) * log(x) + (y-1/2) * log(y)
                          - (x+y-1/2) * log(x+y) + log(2*pi)/2)

    The approximation additionally improves accuracy near zero by iteratively
    shifting the log Gamma approximation using the recursion::

        lgamma(x) = lgamma(x + 1) - log(x)

    If this recursion is applied ``n`` times, then absolute error is bounded by
    ``error < 0.082 / n < tol``, thus we choose ``n`` based on the user
    provided ``tol``.

    :param torch.Tensor x: A positive tensor.
    :param torch.Tensor y: A positive tensor.
    :param float tol: Bound on maximum absolute error. Defaults to 0.1. For
        very small ``tol``, this function simply defers to :func:`log_beta`.
    :rtype: torch.Tensor
    """
    assert isinstance(tol, (float, int)) and tol >= 0
    if tol < 0.02:
        # At small tolerance it is cheaper to defer to torch.lgamma().
        return x.lgamma() + y.lgamma() - (x + y).lgamma()

    # This bound holds for arbitrary x,y. We could do better with large x,y.
    shift = int(math.ceil(0.082 / tol))

    xy = x + y
    factors = []
    for _ in range(shift):
        factors.append(xy / (x * y))
        x = x + 1
        y = y + 1
        xy = xy + 1

    log_factor = functools.reduce(operator.mul, factors).log()

    return (log_factor + (x - 0.5) * x.log() + (y - 0.5) * y.log()
            - (xy - 0.5) * xy.log() + (math.log(2 * math.pi) / 2 - shift))


@torch.no_grad()
def log_binomial(n, k, tol=0.):
    """
    Computes log binomial coefficient.

    When ``tol >= 0.02`` this uses a shifted Stirling's approximation to the
    log Beta function via :func:`log_beta`.

    :param torch.Tensor n: A nonnegative integer tensor.
    :param torch.Tensor k: An integer tensor ranging in ``[0, n]``.
    :rtype: torch.Tensor
    """
    assert isinstance(tol, (float, int)) and tol >= 0
    n_plus_1 = n + 1
    if tol < 0.02:
        # At small tolerance it is cheaper to defer to torch.lgamma().
        return n_plus_1.lgamma() - (k + 1).lgamma() - (n_plus_1 - k).lgamma()

    return -n_plus_1.log() - log_beta(k + 1, n_plus_1 - k, tol=tol)
