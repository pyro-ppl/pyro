# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.autograd import grad

from pyro.util import warn_if_nan
from pyro.ops.linalg import rinverse, eig_3d


def newton_step(loss, x, trust_radius=None):
    """
    Performs a Newton update step to minimize loss on a batch of variables,
    optionally constraining to a trust region [1].

    This is especially usful because the final solution of newton iteration
    is differentiable wrt the inputs, even when all but the final ``x`` is
    detached, due to this method's quadratic convergence [2]. ``loss`` must be
    twice-differentiable as a function of ``x``. If ``loss`` is ``2+d``-times
    differentiable, then the return value of this function is ``d``-times
    differentiable.

    When ``loss`` is interpreted as a negative log probability density, then
    the return values ``mode,cov`` of this function can be used to construct a
    Laplace approximation ``MultivariateNormal(mode,cov)``.

    .. warning:: Take care to detach the result of this function when used in
        an optimization loop. If you forget to detach the result of this
        function during optimization, then backprop will propagate through
        the entire iteration process, and worse will compute two extra
        derivatives for each step.

    Example use inside a loop::

        x = torch.zeros(1000, 2)  # arbitrary initial value
        for step in range(100):
            x = x.detach()          # block gradients through previous steps
            x.requires_grad = True  # ensure loss is differentiable wrt x
            loss = my_loss_function(x)
            x = newton_step(loss, x, trust_radius=1.0)
        # the final x is still differentiable

    [1] Yuan, Ya-xiang. Iciam. Vol. 99. 2000.
        "A review of trust region algorithms for optimization."
        ftp://ftp.cc.ac.cn/pub/yyx/papers/p995.pdf
    [2] Christianson, Bruce. Optimization Methods and Software 3.4 (1994)
        "Reverse accumulation and attractive fixed points."
        http://uhra.herts.ac.uk/bitstream/handle/2299/4338/903839.pdf

    :param torch.Tensor loss: A scalar function of ``x`` to be minimized.
    :param torch.Tensor x: A dependent variable of shape ``(N, D)``
        where ``N`` is the batch size and ``D`` is a small number.
    :param float trust_radius: An optional trust region trust_radius. The
        updated value ``mode`` of this function will be within
        ``trust_radius`` of the input ``x``.
    :return: A pair ``(mode, cov)`` where ``mode`` is an updated tensor
        of the same shape as the original value ``x``, and ``cov`` is an
        esitmate of the covariance DxD matrix with
        ``cov.shape == x.shape[:-1] + (D,D)``.
    :rtype: tuple
    """
    if x.dim() < 1:
        raise ValueError('Expected x to have at least one dimension, actual shape {}'.format(x.shape))
    dim = x.shape[-1]
    if dim == 1:
        return newton_step_1d(loss, x, trust_radius)
    elif dim == 2:
        return newton_step_2d(loss, x, trust_radius)
    elif dim == 3:
        return newton_step_3d(loss, x, trust_radius)
    else:
        raise NotImplementedError('newton_step_nd is not implemented')


def newton_step_1d(loss, x, trust_radius=None):
    """
    Performs a Newton update step to minimize loss on a batch of 1-dimensional
    variables, optionally regularizing to constrain to a trust region.

    See :func:`newton_step` for details.

    :param torch.Tensor loss: A scalar function of ``x`` to be minimized.
    :param torch.Tensor x: A dependent variable with rightmost size of 1.
    :param float trust_radius: An optional trust region trust_radius. The
        updated value ``mode`` of this function will be within
        ``trust_radius`` of the input ``x``.
    :return: A pair ``(mode, cov)`` where ``mode`` is an updated tensor
        of the same shape as the original value ``x``, and ``cov`` is an
        esitmate of the covariance 1x1 matrix with
        ``cov.shape == x.shape[:-1] + (1,1)``.
    :rtype: tuple
    """
    if loss.shape != ():
        raise ValueError('Expected loss to be a scalar, actual shape {}'.format(loss.shape))
    if x.dim() < 1 or x.shape[-1] != 1:
        raise ValueError('Expected x to have rightmost size 1, actual shape {}'.format(x.shape))

    # compute derivatives
    g = grad(loss, [x], create_graph=True)[0]
    H = grad(g.sum(), [x], create_graph=True)[0]
    warn_if_nan(g, 'g')
    warn_if_nan(H, 'H')
    Hinv = H.clamp(min=1e-8).reciprocal()
    dx = -g * Hinv
    dx[~(dx == dx)] = 0
    if trust_radius is not None:
        dx.clamp_(min=-trust_radius, max=trust_radius)

    # apply update
    x_new = x.detach() + dx
    assert x_new.shape == x.shape
    return x_new, Hinv.unsqueeze(-1)


def newton_step_2d(loss, x, trust_radius=None):
    """
    Performs a Newton update step to minimize loss on a batch of 2-dimensional
    variables, optionally regularizing to constrain to a trust region.

    See :func:`newton_step` for details.

    :param torch.Tensor loss: A scalar function of ``x`` to be minimized.
    :param torch.Tensor x: A dependent variable with rightmost size of 2.
    :param float trust_radius: An optional trust region trust_radius. The
        updated value ``mode`` of this function will be within
        ``trust_radius`` of the input ``x``.
    :return: A pair ``(mode, cov)`` where ``mode`` is an updated tensor
        of the same shape as the original value ``x``, and ``cov`` is an
        esitmate of the covariance 2x2 matrix with
        ``cov.shape == x.shape[:-1] + (2,2)``.
    :rtype: tuple
    """
    if loss.shape != ():
        raise ValueError('Expected loss to be a scalar, actual shape {}'.format(loss.shape))
    if x.dim() < 1 or x.shape[-1] != 2:
        raise ValueError('Expected x to have rightmost size 2, actual shape {}'.format(x.shape))

    # compute derivatives
    g = grad(loss, [x], create_graph=True)[0]
    H = torch.stack([grad(g[..., 0].sum(), [x], create_graph=True)[0],
                     grad(g[..., 1].sum(), [x], create_graph=True)[0]], -1)
    assert g.shape[-1:] == (2,)
    assert H.shape[-2:] == (2, 2)
    warn_if_nan(g, 'g')
    warn_if_nan(H, 'H')

    if trust_radius is not None:
        # regularize to keep update within ball of given trust_radius
        detH = H[..., 0, 0] * H[..., 1, 1] - H[..., 0, 1] * H[..., 1, 0]
        mean_eig = (H[..., 0, 0] + H[..., 1, 1]) / 2
        min_eig = mean_eig - (mean_eig ** 2 - detH).clamp(min=0).sqrt()
        regularizer = (g.pow(2).sum(-1).sqrt() / trust_radius - min_eig).clamp_(min=1e-8)
        warn_if_nan(regularizer, 'regularizer')
        H = H + regularizer.unsqueeze(-1).unsqueeze(-1) * torch.eye(2, dtype=H.dtype, device=H.device)

    # compute newton update
    Hinv = rinverse(H, sym=True)
    warn_if_nan(Hinv, 'Hinv')

    # apply update
    x_new = x.detach() - (Hinv * g.unsqueeze(-2)).sum(-1)
    assert x_new.shape == x.shape
    return x_new, Hinv


def newton_step_3d(loss, x, trust_radius=None):
    """
    Performs a Newton update step to minimize loss on a batch of 3-dimensional
    variables, optionally regularizing to constrain to a trust region.

    See :func:`newton_step` for details.

    :param torch.Tensor loss: A scalar function of ``x`` to be minimized.
    :param torch.Tensor x: A dependent variable with rightmost size of 2.
    :param float trust_radius: An optional trust region trust_radius. The
        updated value ``mode`` of this function will be within
        ``trust_radius`` of the input ``x``.
    :return: A pair ``(mode, cov)`` where ``mode`` is an updated tensor
        of the same shape as the original value ``x``, and ``cov`` is an
        esitmate of the covariance 3x3 matrix with
        ``cov.shape == x.shape[:-1] + (3,3)``.
    :rtype: tuple
    """
    if loss.shape != ():
        raise ValueError('Expected loss to be a scalar, actual shape {}'.format(loss.shape))
    if x.dim() < 1 or x.shape[-1] != 3:
        raise ValueError('Expected x to have rightmost size 3, actual shape {}'.format(x.shape))

    # compute derivatives
    g = grad(loss, [x], create_graph=True)[0]
    H = torch.stack([grad(g[..., 0].sum(), [x], create_graph=True)[0],
                     grad(g[..., 1].sum(), [x], create_graph=True)[0],
                     grad(g[..., 2].sum(), [x], create_graph=True)[0]], -1)
    assert g.shape[-1:] == (3,)
    assert H.shape[-2:] == (3, 3)
    warn_if_nan(g, 'g')
    warn_if_nan(H, 'H')

    if trust_radius is not None:
        # regularize to keep update within ball of given trust_radius
        # calculate eigenvalues of symmetric matrix
        min_eig, _, _ = eig_3d(H)
        regularizer = (g.pow(2).sum(-1).sqrt() / trust_radius - min_eig).clamp_(min=1e-8)
        warn_if_nan(regularizer, 'regularizer')
        H = H + regularizer.unsqueeze(-1).unsqueeze(-1) * torch.eye(3, dtype=H.dtype, device=H.device)

    # compute newton update
    Hinv = rinverse(H, sym=True)
    warn_if_nan(Hinv, 'Hinv')

    # apply update
    x_new = x.detach() - (Hinv * g.unsqueeze(-2)).sum(-1)
    assert x_new.shape == x.shape, "{} {}".format(x_new.shape, x.shape)
    return x_new, Hinv
