from __future__ import absolute_import, division, print_function

import warnings

import torch
from torch.autograd import grad


def _warn_if_nan(tensor, name):
    if torch.isnan(tensor).any():
        warnings.warn('Encountered nan elements in {}'.format(name))


def newton_step_2d(loss, x, trust_radius=None):
    """
    Performs a Newton update step to minimize loss on a batch of 2-dimensional
    variables, optionally regularizing to constrain to a trust region.

    ``loss`` must be twice-differentiable as a function of ``x``. If ``loss``
    is ``2+d``-times differentiable, then the return value of this function is
    ``d``-times differentiable.

    When ``loss`` is interpreted as a negative log probability density, then
    the return value of this function can be used to construct a Laplace
    approximation ``MultivariateNormal(mode,cov)``.

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
            x = newton_step_2d(loss, x, trust_radius=1.0)
        # the final x is still differentiable

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
    _warn_if_nan(g, 'g')
    _warn_if_nan(H, 'H')

    if trust_radius is not None:
        # regularize to keep update within ball of given trust_radius
        detH = H[..., 0, 0] * H[..., 1, 1] - H[..., 0, 1] * H[..., 1, 0]
        mean_eig = (H[..., 0, 0] + H[..., 1, 1]) / 2
        min_eig = mean_eig - (mean_eig ** 2 - detH).sqrt()
        regularizer = (g.pow(2).sum(-1).sqrt() / trust_radius - min_eig).clamp_(min=1e-8)
        _warn_if_nan(regularizer, 'regularizer')
        H = H + regularizer.unsqueeze(-1).unsqueeze(-1) * H.new([[1.0, 0.0], [0.0, 1.0]])

    # compute newton update
    detH = H[..., 0, 0] * H[..., 1, 1] - H[..., 0, 1] * H[..., 1, 0]
    Hinv = H.new(H.shape)
    Hinv[..., 0, 0] = H[..., 1, 1]
    Hinv[..., 0, 1] = -H[..., 0, 1]
    Hinv[..., 1, 0] = -H[..., 1, 0]
    Hinv[..., 1, 1] = H[..., 0, 0]
    Hinv = Hinv / detH.unsqueeze(-1).unsqueeze(-1)
    _warn_if_nan(Hinv, 'Hinv')

    # apply update
    x_new = x.detach() - (Hinv * g.unsqueeze(-2)).sum(-1)
    assert x_new.shape == x.shape
    return x_new, Hinv
