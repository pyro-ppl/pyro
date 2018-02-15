from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch.multivariate_normal import MultivariateNormal


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
    support = constraints.real
    has_rsample = True
    reparameterized = True

    def __init__(self, loc, scale_tril):
        self.loc = loc
        self.scale_tril = scale_tril
        assert(len(loc.size()) == 1), "OMTMultivariateNormal loc must be 1-dimensional"
        assert(len(scale_tril.size()) == 2), "OMTMultivariateNormal scale_tril must be 2-dimensional"

    @property
    def variance(self):
        return self.covariance_matrix.diag()

    @lazy_property
    def covariance_matrix(self):
        return torch.mm(self.scale_tril, self.scale_tril.t())

    def rsample(self, sample_shape=torch.Size()):
        return _OMTMVNSample.apply(self.loc, self.scale_tril, sample_shape + self.loc.shape)


def sum_all_but_rightmost_dim(x):
    if len(x.size()) == 1:
        return x
    return x.view(-1, x.size(-1)).sum(0)


def sum_all_but_rightmost_two_dims(x):
    if len(x.size()) == 2:
        return x
    return x.view(-1, x.size(-2), x.size(-1)).sum(0)


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
        loc_grad = sum_all_but_rightmost_dim(grad_output)

        z_ja = z.unsqueeze(-1)
        g_transpose = torch.transpose(g, 0, -1)
        g_ij = g_transpose.contiguous().view(dim, -1)
        L_inv_bi_g_ji = torch.trtrs(g_ij, L, upper=False)[0].view(g_transpose.size())
        L_inv_bi_g_ji = torch.transpose(L_inv_bi_g_ji, 0, -1).unsqueeze(-2)
        diff_L_ab = 0.5 * sum_all_but_rightmost_two_dims(L_inv_bi_g_ji * z_ja)

        epsilon_jb = epsilon.unsqueeze(-2)
        g_ja = g.unsqueeze(-1)
        diff_L_ab += 0.5 * sum_all_but_rightmost_two_dims(g_ja * epsilon_jb)

        identity = torch.eye(dim).type_as(g)
        R_inv = torch.trtrs(identity, L, transpose=True, upper=False)[0]
        Sigma_inv = torch.mm(R_inv, R_inv.t())
        V, D, Vt = torch.svd(Sigma_inv + jitter)
        D_outer = D.unsqueeze(-1) + D.unsqueeze(0) + jitter

        # can i do this expansion more cleanly?
        jdim = len(z.size()) - 1
        expand_tuple = tuple([-1]*jdim + [dim, dim])
        z_tilde = identity * torch.matmul(z, Vt).unsqueeze(-1).expand(*expand_tuple)
        g_tilde = identity * torch.matmul(g, Vt).unsqueeze(-1).expand(*expand_tuple)

        Y = torch.matmul(z_tilde, torch.matmul(1.0 / D_outer, g_tilde))
        Y = sum_all_but_rightmost_two_dims(Y)
        Y_tilde = torch.mm(V, torch.mm(Y, Vt))
        Y_tilde_symm = Y_tilde + torch.transpose(Y_tilde, 0, 1)

        Tr_xi_1_Y = 0.5 * torch.mm(Sigma_inv, Y_tilde_symm)
        Tr_xi_1_Y = torch.mm(Tr_xi_1_Y, R_inv)
        Tr_xi_2_Y = -0.5 * torch.mm(torch.mm(Y_tilde_symm, Sigma_inv), R_inv)
        diff_L_ab += Tr_xi_1_Y + Tr_xi_2_Y
        L_grad = torch.tril(diff_L_ab)

        return loc_grad, L_grad, None
