# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import operator


def log_beta(x, y):
    """
    Log Beta function.

    :param torch.Tensor x: A positive tensor.
    :param torch.Tensor y: A positive tensor.
    """
    return x.lgamma() + y.lgamma() - (x + y).lgamma()


def log_beta_stirling(x, y, tol=0.1):
    """
    Shifted Stirling's approximation to the log Beta function.

    This is useful as a cheaper alternative to :func:`log_beta`.

    This adapts Stirling's approximation of the log Gamma function::

        lgamma(z) ≈ (z - 1/2) * log(z) - z + log(2 * pi) / 2

    to approximate the log Beta function::

        log_beta(x, y) ≈ ((x-1/2) * log(x) + (y-1/2) * log(y)
                          - (x+y-1/2) * log(x+y) + log(2*pi)/2)

    This additionally improves accuracy near zero by iteratively shifting
    the log Gamma approximation using the recursion::

        lgamma(x) = lgamma(x + 1) - log(x)

    If this recursion is applied ``n`` times, then absolute error is bounded by
    ``error < 0.082 / n < tol``, thus we choose ``n`` based on the user
    provided ``tol``.

    :param torch.Tensor x: A positive tensor.
    :param torch.Tensor y: A positive tensor.
    :param float tol: Bound on maximum absolute error. Defaults to 0.1.
        For very small ``tol``, this function simply defers to :func:`log_beta`.
    """
    assert isinstance(tol, (float, int)) and tol >= 0
    if tol < 0.02:
        # Eventually it is cheaper to defer to torch.lgamma().
        return log_beta(x, y)

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
