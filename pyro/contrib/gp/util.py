# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.infer import TraceMeanField_ELBO
from pyro.infer.util import torch_backward, torch_item


def conditional(Xnew, X, kernel, f_loc, f_scale_tril=None, Lff=None, full_cov=False,
                whiten=False, jitter=1e-6):
    r"""
    Given :math:`X_{new}`, predicts loc and covariance matrix of the conditional
    multivariate normal distribution

    .. math:: p(f^*(X_{new}) \mid X, k, f_{loc}, f_{scale\_tril}).

    Here ``f_loc`` and ``f_scale_tril`` are variation parameters of the variational
    distribution

    .. math:: q(f \mid f_{loc}, f_{scale\_tril}) \sim p(f | X, y),

    where :math:`f` is the function value of the Gaussian Process given input :math:`X`

    .. math:: p(f(X)) \sim \mathcal{N}(0, k(X, X))

    and :math:`y` is computed from :math:`f` by some likelihood function
    :math:`p(y|f)`.

    In case ``f_scale_tril=None``, we consider :math:`f = f_{loc}` and computes

    .. math:: p(f^*(X_{new}) \mid X, k, f).

    In case ``f_scale_tril`` is not ``None``, we follow the derivation from reference
    [1]. For the case ``f_scale_tril=None``, we follow the popular reference [2].

    References:

    [1] `Sparse GPs: approximate the posterior, not the model
    <https://www.prowler.io/sparse-gps-approximate-the-posterior-not-the-model/>`_

    [2] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor Xnew: A new input data.
    :param torch.Tensor X: An input data to be conditioned on.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor f_loc: Mean of :math:`q(f)`. In case ``f_scale_tril=None``,
        :math:`f_{loc} = f`.
    :param torch.Tensor f_scale_tril: Lower triangular decomposition of covariance
        matrix of :math:`q(f)`'s .
    :param torch.Tensor Lff: Lower triangular decomposition of :math:`kernel(X, X)`
        (optional).
    :param bool full_cov: A flag to decide if we want to return full covariance
        matrix or just variance.
    :param bool whiten: A flag to tell if ``f_loc`` and ``f_scale_tril`` are
        already transformed by the inverse of ``Lff``.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
    # p(f* | Xnew, X, kernel, f_loc, f_scale_tril) ~ N(f* | loc, cov)
    # Kff = Lff @ Lff.T
    # v = inv(Lff) @ f_loc  <- whitened f_loc
    # S = inv(Lff) @ f_scale_tril  <- whitened f_scale_tril
    # Denote:
    #     W = (inv(Lff) @ Kf*).T
    #     K = W @ S @ S.T @ W.T
    #     Q** = K*f @ inv(Kff) @ Kf* = W @ W.T
    # loc = K*f @ inv(Kff) @ f_loc = W @ v
    # Case 1: f_scale_tril = None
    #     cov = K** - K*f @ inv(Kff) @ Kf* = K** - Q**
    # Case 2: f_scale_tril != None
    #     cov = K** - Q** + K*f @ inv(Kff) @ f_cov @ inv(Kff) @ Kf*
    #         = K** - Q** + W @ S @ S.T @ W.T
    #         = K** - Q** + K

    N = X.size(0)
    M = Xnew.size(0)
    latent_shape = f_loc.shape[:-1]

    if Lff is None:
        Kff = kernel(X).contiguous()
        Kff.view(-1)[::N + 1] += jitter  # add jitter to diagonal
        Lff = Kff.cholesky()
    Kfs = kernel(X, Xnew)

    # convert f_loc_shape from latent_shape x N to N x latent_shape
    f_loc = f_loc.permute(-1, *range(len(latent_shape)))
    # convert f_loc to 2D tensor for packing
    f_loc_2D = f_loc.reshape(N, -1)
    if f_scale_tril is not None:
        # convert f_scale_tril_shape from latent_shape x N x N to N x N x latent_shape
        f_scale_tril = f_scale_tril.permute(-2, -1, *range(len(latent_shape)))
        # convert f_scale_tril to 2D tensor for packing
        f_scale_tril_2D = f_scale_tril.reshape(N, -1)

    if whiten:
        v_2D = f_loc_2D
        W = Kfs.triangular_solve(Lff, upper=False)[0].t()
        if f_scale_tril is not None:
            S_2D = f_scale_tril_2D
    else:
        pack = torch.cat((f_loc_2D, Kfs), dim=1)
        if f_scale_tril is not None:
            pack = torch.cat((pack, f_scale_tril_2D), dim=1)

        Lffinv_pack = pack.triangular_solve(Lff, upper=False)[0]
        # unpack
        v_2D = Lffinv_pack[:, :f_loc_2D.size(1)]
        W = Lffinv_pack[:, f_loc_2D.size(1):f_loc_2D.size(1) + M].t()
        if f_scale_tril is not None:
            S_2D = Lffinv_pack[:, -f_scale_tril_2D.size(1):]

    loc_shape = latent_shape + (M,)
    loc = W.matmul(v_2D).t().reshape(loc_shape)

    if full_cov:
        Kss = kernel(Xnew)
        Qss = W.matmul(W.t())
        cov = Kss - Qss
    else:
        Kssdiag = kernel(Xnew, diag=True)
        Qssdiag = W.pow(2).sum(dim=-1)
        # Theoretically, Kss - Qss is non-negative; but due to numerical
        # computation, that might not be the case in practice.
        var = (Kssdiag - Qssdiag).clamp(min=0)

    if f_scale_tril is not None:
        W_S_shape = (Xnew.size(0),) + f_scale_tril.shape[1:]
        W_S = W.matmul(S_2D).reshape(W_S_shape)
        # convert W_S_shape from M x N x latent_shape to latent_shape x M x N
        W_S = W_S.permute(list(range(2, W_S.dim())) + [0, 1])

        if full_cov:
            St_Wt = W_S.transpose(-2, -1)
            K = W_S.matmul(St_Wt)
            cov = cov + K
        else:
            Kdiag = W_S.pow(2).sum(dim=-1)
            var = var + Kdiag
    else:
        if full_cov:
            cov = cov.expand(latent_shape + (M, M))
        else:
            var = var.expand(latent_shape + (M,))

    return (loc, cov) if full_cov else (loc, var)


def train(gpmodule, optimizer=None, loss_fn=None, retain_graph=None, num_steps=1000):
    """
    A helper to optimize parameters for a GP module.

    :param ~pyro.contrib.gp.models.GPModel gpmodule: A GP module.
    :param ~torch.optim.Optimizer optimizer: A PyTorch optimizer instance.
        By default, we use Adam with ``lr=0.01``.
    :param callable loss_fn: A loss function which takes inputs are
        ``gpmodule.model``, ``gpmodule.guide``, and returns ELBO loss.
        By default, ``loss_fn=TraceMeanField_ELBO().differentiable_loss``.
    :param bool retain_graph: An optional flag of ``torch.autograd.backward``.
    :param int num_steps: Number of steps to run SVI.
    :returns: a list of losses during the training procedure
    :rtype: list
    """
    optimizer = (torch.optim.Adam(gpmodule.parameters(), lr=0.01)
                 if optimizer is None else optimizer)
    # TODO: add support for JIT loss
    loss_fn = TraceMeanField_ELBO().differentiable_loss if loss_fn is None else loss_fn

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(gpmodule.model, gpmodule.guide)
        torch_backward(loss, retain_graph)
        return loss

    losses = []
    for i in range(num_steps):
        loss = optimizer.step(closure)
        losses.append(torch_item(loss))
    return losses
