from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints

from pyro.distributions.torch.multivariate_normal import MultivariateNormal
from pyro.distributions.util import sum_leftmost


class OMTMultivariateNormal(MultivariateNormal):
    """Multivariate normal (Gaussian) distribution with OMT gradients w.r.t. both
    parameters. Note the gradient computation w.r.t. the Cholesky factor has cost
    O(D^3), although the resulting gradient variance is generally expected to be lower.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.autograd.Variable loc: Mean.
    :param torch.autograd.Variable scale_tril: Cholesky of Covariance matrix.
    """
    params = {"loc": constraints.real, "scale_tril": constraints.lower_triangular}

    def __init__(self, loc, scale_tril):
        assert(loc.dim() == 1), "OMTMultivariateNormal loc must be 1-dimensional"
        assert(scale_tril.dim() == 2), "OMTMultivariateNormal scale_tril must be 2-dimensional"
        covariance_matrix = torch.mm(scale_tril, scale_tril.t())
        super(OMTMultivariateNormal, self).__init__(loc, covariance_matrix)
        self.scale_tril = scale_tril

    def rsample(self, sample_shape=torch.Size()):
        return _OMTMVNSample.apply(self.loc, self.scale_tril, sample_shape + self.loc.shape)


class _OMTMVNSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_tril, shape):
        ctx.white = loc.new(shape).normal_()
        ctx.z = torch.matmul(ctx.white, scale_tril.t())
        ctx.save_for_backward(scale_tril)
        return loc + ctx.z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        jitter = 1.0e-12  # do i really need this?
        L, = ctx.saved_tensors
        z = ctx.z
        epsilon = ctx.white

        dim = L.size(0)
        g = grad_output
        loc_grad = sum_leftmost(grad_output, -1)

        z_ja = z.unsqueeze(-1)
        g_transpose = torch.transpose(g, 0, -1)
        g_ij = g_transpose.contiguous().view(dim, -1)
        L_inv_bi_g_ji = torch.trtrs(g_ij, L, upper=False)[0].view(g_transpose.size())
        L_inv_bi_g_ji = torch.transpose(L_inv_bi_g_ji, 0, -1).unsqueeze(-2)
        diff_L_ab = 0.5 * sum_leftmost(L_inv_bi_g_ji * z_ja, -2)

        epsilon_jb = epsilon.unsqueeze(-2)
        g_ja = g.unsqueeze(-1)
        diff_L_ab += 0.5 * sum_leftmost(g_ja * epsilon_jb, -2)

        identity = torch.eye(dim).type_as(g)
        R_inv = torch.trtrs(identity, L, transpose=True, upper=False)[0]
        Sigma_inv = torch.mm(R_inv, R_inv.t())
        V, D, Vt = torch.svd(Sigma_inv + jitter)
        D_outer = D.unsqueeze(-1) + D.unsqueeze(0) + jitter

        # XXX can i do this expansion more cleanly?
        jdim = len(z.size()) - 1
        expand_tuple = tuple([-1]*jdim + [dim, dim])
        z_tilde = identity * torch.matmul(z, Vt).unsqueeze(-1).expand(*expand_tuple)
        g_tilde = identity * torch.matmul(g, Vt).unsqueeze(-1).expand(*expand_tuple)

        Y = torch.matmul(z_tilde, torch.matmul(1.0 / D_outer, g_tilde))
        Y = sum_leftmost(Y, -2)
        Y_tilde = torch.mm(V, torch.mm(Y, Vt))
        Y_tilde_symm = Y_tilde + torch.transpose(Y_tilde, 0, 1)

        Tr_xi_1_Y = 0.5 * torch.mm(Sigma_inv, Y_tilde_symm)
        Tr_xi_1_Y = torch.mm(Tr_xi_1_Y, R_inv)
        Tr_xi_2_Y = -0.5 * torch.mm(torch.mm(Y_tilde_symm, Sigma_inv), R_inv)
        diff_L_ab += Tr_xi_1_Y + Tr_xi_2_Y
        L_grad = torch.tril(diff_L_ab)

        return loc_grad, L_grad, None
