# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints

from pyro.distributions.torch import MultivariateNormal
from pyro.distributions.util import eye_like, sum_leftmost


class OMTMultivariateNormal(MultivariateNormal):
    """Multivariate normal (Gaussian) distribution with OMT gradients w.r.t. both
    parameters. Note the gradient computation w.r.t. the Cholesky factor has cost
    O(D^3), although the resulting gradient variance is generally expected to be lower.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.Tensor loc: Mean.
    :param torch.Tensor scale_tril: Cholesky of Covariance matrix.
    """
    arg_constraints = {"loc": constraints.real, "scale_tril": constraints.lower_triangular}

    def __init__(self, loc, scale_tril):
        if loc.dim() != 1:
            raise ValueError("OMTMultivariateNormal loc must be 1-dimensional")
        if scale_tril.dim() != 2:
            raise ValueError("OMTMultivariateNormal scale_tril must be 2-dimensional")
        super().__init__(loc, scale_tril=scale_tril)

    def rsample(self, sample_shape=torch.Size()):
        return _OMTMVNSample.apply(self.loc, self.scale_tril, sample_shape + self.loc.shape)


class _OMTMVNSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_tril, shape):
        white = torch.randn(shape, dtype=loc.dtype, device=loc.device)
        z = torch.matmul(white, scale_tril.t())
        ctx.save_for_backward(z, white, scale_tril)
        return loc + z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        jitter = 1.0e-8  # do i really need this?
        z, epsilon, L = ctx.saved_tensors

        dim = L.shape[0]
        g = grad_output
        loc_grad = sum_leftmost(grad_output, -1)

        identity = eye_like(g, dim)
        R_inv = torch.triangular_solve(identity, L.t(), transpose=False, upper=True)[0]

        z_ja = z.unsqueeze(-1)
        g_R_inv = torch.matmul(g, R_inv).unsqueeze(-2)
        epsilon_jb = epsilon.unsqueeze(-2)
        g_ja = g.unsqueeze(-1)
        diff_L_ab = 0.5 * sum_leftmost(g_ja * epsilon_jb + g_R_inv * z_ja, -2)

        Sigma_inv = torch.mm(R_inv, R_inv.t())
        V, D, _ = torch.svd(Sigma_inv + jitter)
        D_outer = D.unsqueeze(-1) + D.unsqueeze(0)

        expand_tuple = tuple([-1] * (z.dim() - 1) + [dim, dim])
        z_tilde = identity * torch.matmul(z, V).unsqueeze(-1).expand(*expand_tuple)
        g_tilde = identity * torch.matmul(g, V).unsqueeze(-1).expand(*expand_tuple)

        Y = sum_leftmost(torch.matmul(z_tilde, torch.matmul(1.0 / D_outer, g_tilde)), -2)
        Y = torch.mm(V, torch.mm(Y, V.t()))
        Y = Y + Y.t()

        Tr_xi_Y = torch.mm(torch.mm(Sigma_inv, Y), R_inv) - torch.mm(Y, torch.mm(Sigma_inv, R_inv))
        diff_L_ab += 0.5 * Tr_xi_Y
        L_grad = torch.tril(diff_L_ab)

        return loc_grad, L_grad, None
