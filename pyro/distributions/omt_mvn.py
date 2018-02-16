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

        identity = torch.eye(dim).type_as(g)
        R_inv = torch.trtrs(identity, L.t(), transpose=False, upper=True)[0]

        z_ja = z.unsqueeze(-1)
        g_R_inv = torch.matmul(g, R_inv).unsqueeze(-2)
        diff_L_ab = 0.5 * sum_leftmost(g_R_inv * z_ja, -2)

        epsilon_jb = epsilon.unsqueeze(-2)
        g_ja = g.unsqueeze(-1)
        diff_L_ab += 0.5 * sum_leftmost(g_ja * epsilon_jb, -2)

        Sigma_inv = torch.mm(R_inv, R_inv.t())
        V, D, Vt = torch.svd(Sigma_inv + jitter)
        D_outer = D.unsqueeze(-1) + D.unsqueeze(0) + jitter

        # XXX can i do this expansion more cleanly?
        jdim = z.dim() - 1
        expand_tuple = tuple([-1]*jdim + [dim, dim])
        z_tilde = identity * torch.matmul(z, V).unsqueeze(-1).expand(*expand_tuple)
        g_tilde = identity * torch.matmul(g, V).unsqueeze(-1).expand(*expand_tuple)

        Y = torch.matmul(z_tilde, torch.matmul(1.0 / D_outer, g_tilde))
        Y_tilde = torch.matmul(V, torch.matmul(Y, Vt))
        Y_tilde_symm = Y_tilde + torch.transpose(Y_tilde, -1, -2)

        Tr_xi_Y = torch.matmul(torch.matmul(Sigma_inv, Y_tilde_symm), R_inv)
        Tr_xi_Y -= torch.matmul(Y_tilde_symm, torch.mm(Sigma_inv, R_inv))
        diff_L_ab += 0.5 * sum_leftmost(Tr_xi_Y, -2)
        L_grad = torch.tril(diff_L_ab)

        return loc_grad, L_grad, None


class OTCVMultivariateNormal(MultivariateNormal):
    """Multivariate normal (Gaussian) distribution with optimal transport-inspired control variates.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.autograd.Variable loc: Mean.
    :param torch.autograd.Variable scale_tril: Cholesky of Covariance matrix.
    :param torch.autograd.Variable B: tensor controlling the control variate
    :param torch.autograd.Variable C: tensor controlling the control variate
    :param torch.autograd.Variable D: tensor controlling the control variate
    :param torch.autograd.Variable F: tensor controlling the control variate
    """
    params = {"loc": constraints.real, "scale_tril": constraints.lower_triangular}

    def __init__(self, loc, scale_tril, B=None, C=None, D=None, F=None):
        assert(loc.dim() == 1), "OMTMultivariateNormal loc must be 1-dimensional"
        assert(scale_tril.dim() == 2), "OMTMultivariateNormal scale_tril must be 2-dimensional"
        covariance_matrix = torch.mm(scale_tril, scale_tril.t())
        super(OTCVMultivariateNormal, self).__init__(loc, covariance_matrix)
        self.scale_tril = scale_tril
        self.B = B
        self.C = C
        self.D = D
        self.F = F
        BC_mode = (B is not None) and (C is not None)
        DF_mode = (D is not None) and (F is not None)
        if BC_mode:
            assert(scale_tril.size() == B.size() == C.size())
        if DF_mode:
            assert(scale_tril.size() == D.size() == F.size())
        assert (BC_mode or DF_mode), "Must use at least one control variate parameterization"

    def rsample(self, sample_shape=torch.Size()):
        return _OTCVMVNSample.apply(self.loc, self.scale_tril, self.B, self.C, self.D, self.F,
                                    sample_shape + self.loc.shape)


class _OTCVMVNSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_tril, B, C, D, F, shape):
        ctx.save_for_backward(scale_tril)
        ctx.white = loc.new(shape).normal_()
        ctx.z = torch.matmul(ctx.white, scale_tril.t())
        ctx.B, ctx.C = B, C
        ctx.D, ctx.F = D, F
        return loc + ctx.z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        L, = ctx.saved_tensors
        z = ctx.z
        epsilon = ctx.white
        B, C = ctx.B, ctx.C
        D, F = ctx.D, ctx.F
        BC_mode = (B is not None) and (C is not None)
        DF_mode = (D is not None) and (F is not None)

        dim = L.size(0)
        g = grad_output
        loc_grad = sum_leftmost(grad_output, -1)

        # compute the rep trick gradient
        epsilon_jb = epsilon.unsqueeze(-2)
        g_ja = g.unsqueeze(-1)
        diff_L_ab = sum_leftmost(g_ja * epsilon_jb, -2)

        # modulate the velocity field with an infinitessimal rotation
        if BC_mode:
            LB = torch.mm(L, B)
            eps_C = torch.matmul(epsilon, C)
            g_LB = torch.matmul(g, LB)
            diff_L_ab += sum_leftmost(eps_C.unsqueeze(-2) * g_LB.unsqueeze(-1), -2)
            LC = torch.mm(L, C)
            eps_B = torch.matmul(epsilon, B)
            g_LC = torch.matmul(g, LC)
            diff_L_ab -= sum_leftmost(eps_B.unsqueeze(-1) * g_LC.unsqueeze(-2), -2)

        # modulate the velocity field with an infinitessimal rotation
        if DF_mode:
            LD = torch.mm(L, D)
            g_LD = torch.matmul(g, LD)
            multiplier = (g_LD * epsilon).sum()
            diff_L_ab += multiplier * F

        L_grad = torch.tril(diff_L_ab)

        return loc_grad, L_grad, None, None, None, None, None
