# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import operator
import weakref
from typing import Dict

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss


class _SafeLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.log()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / x.clamp(min=torch.finfo(x.dtype).eps)


def safe_log(x):
    """
    Like :func:`torch.log` but avoids infinite gradients at log(0)
    by clamping them to at most ``1 / finfo.eps``.
    """
    return _SafeLog.apply(x)


def log_beta(x, y, tol=0.0):
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

    return (
        log_factor
        + (x - 0.5) * x.log()
        + (y - 0.5) * y.log()
        - (xy - 0.5) * xy.log()
        + (math.log(2 * math.pi) / 2 - shift)
    )


@torch.no_grad()
def log_binomial(n, k, tol=0.0):
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


def log_I1(orders: int, value: torch.Tensor, terms=250):
    r"""Compute first n log modified bessel function of first kind
    .. math ::

        \log(I_v(z)) = v*\log(z/2) + \log(\sum_{k=0}^\inf \exp\left[2*k*\log(z/2) - \sum_kk^k log(kk)
        - \lgamma(v + k + 1)\right])

    :param orders: orders of the log modified bessel function.
    :param value: values to compute modified bessel function for
    :param terms: truncation of summation
    :return: 0 to orders modified bessel function
    """
    orders = orders + 1
    if len(value.size()) == 0:
        vshape = torch.Size([1])
    else:
        vshape = value.shape
    value = value.reshape(-1, 1)

    k = torch.arange(terms, device=value.device)
    lgammas_all = torch.lgamma(torch.arange(1, terms + orders + 1, device=value.device))
    assert lgammas_all.shape == (orders + terms,)  # lgamma(0) = inf => start from 1

    lvalues = torch.log(value / 2) * k.view(1, -1)
    assert lvalues.shape == (vshape.numel(), terms)

    lfactorials = lgammas_all[:terms]
    assert lfactorials.shape == (terms,)

    lgammas = lgammas_all.repeat(orders).view(orders, -1)
    assert lgammas.shape == (orders, terms + orders)  # lgamma(0) = inf => start from 1

    indices = k[:orders].view(-1, 1) + k.view(1, -1)
    assert indices.shape == (orders, terms)

    seqs = (
        2 * lvalues[None, :, :]
        - lfactorials[None, None, :]
        - lgammas.gather(1, indices)[:, None, :]
    ).logsumexp(-1)
    assert seqs.shape == (orders, vshape.numel())

    i1s = lvalues[..., :orders].T + seqs
    assert i1s.shape == (orders, vshape.numel())
    return i1s.view(-1, *vshape)


def get_quad_rule(num_quad, prototype_tensor):
    r"""
    Get quadrature points and corresponding log weights for a Gauss Hermite quadrature rule
    with the specified number of quadrature points.

    Example usage::

        quad_points, log_weights = get_quad_rule(32, prototype_tensor)
        # transform to N(0, 4.0) Normal distribution
        quad_points *= 4.0
        # compute variance integral in log-space using logsumexp and exponentiate
        variance = torch.logsumexp(quad_points.pow(2.0).log() + log_weights, axis=0).exp()
        assert (variance - 16.0).abs().item() < 1.0e-6

    :param int num_quad: number of quadrature points.
    :param torch.Tensor prototype_tensor: used to determine `dtype` and `device` of returned tensors.
    :return: tuple of `torch.Tensor`s of the form `(quad_points, log_weights)`
    """
    quad_rule = hermgauss(num_quad)
    quad_points = quad_rule[0] * np.sqrt(2.0)
    log_weights = np.log(quad_rule[1]) - 0.5 * np.log(np.pi)
    return torch.from_numpy(quad_points).type_as(prototype_tensor), torch.from_numpy(
        log_weights
    ).type_as(prototype_tensor)


def sparse_multinomial_likelihood(total_count, nonzero_logits, nonzero_value):
    """
    The following are equivalent::

        # Version 1. dense
        log_prob = Multinomial(logits=logits).log_prob(value).sum()

        # Version 2. sparse
        nnz = value.nonzero(as_tuple=True)
        log_prob = sparse_multinomial_likelihood(
            value.sum(-1),
            (logits - logits.logsumexp(-1))[nnz],
            value[nnz],
        )
    """
    return (
        _log_factorial_sum(total_count)
        - _log_factorial_sum(nonzero_value)
        + torch.dot(nonzero_logits, nonzero_value)
    )


_log_factorial_cache: Dict[int, torch.Tensor] = {}


def _log_factorial_sum(x: torch.Tensor) -> torch.Tensor:
    if x.requires_grad:
        return (x + 1).lgamma().sum()
    key = id(x)
    if key not in _log_factorial_cache:
        weakref.finalize(x, _log_factorial_cache.pop, key, None)  # type: ignore
        _log_factorial_cache[key] = (x + 1).lgamma().sum()
    return _log_factorial_cache[key]
