from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints

from pyro.distributions.torch import MultivariateNormal
from pyro.distributions.util import sum_leftmost


class AVFMultivariateNormal(MultivariateNormal):
    """Multivariate normal (Gaussian) distribution with transport equation inspired control variates.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.Tensor loc: D-dimensional mean vector.
    :param torch.Tensor scale_tril: Cholesky of Covariance matrix; D x D matrix.
    :param torch.Tensor CV: 2 x L x D tensor that parameterizes the control variate; L is an arbitrary positive integer.

    Example usage::

    CV = torch.tensor(0.1 * torch.ones(2, 1, D), requires_grad=True)
    opt_cv = pyro.optim.Adam([CV], lr=0.1, betas=(0.5, 0.999))

    for _ in range(1000):
        d = AVFMultivariateNormal(loc, scale_tril, CV)
        z = d.rsample(sample_shape=torch.Size([n_samples]))
        cost = torch.pow(z, 2.0).sum()
        cost.backward()
        opt_cv.step()
        opt_cv.zero_grad()

    """
    arg_constraints = {"loc": constraints.real, "scale_tril": constraints.lower_triangular, "CV": constraints.real}

    def __init__(self, loc, scale_tril, CV):
        assert(loc.dim() == 1), "AVFMultivariateNormal loc must be 1-dimensional"
        assert(scale_tril.dim() == 2), "AVFMultivariateNormal scale_tril must be 2-dimensional"
        assert CV.dim() == 3, "CV should be of size 2 x L x D, where D is the dimension of the location parameter loc"
        assert CV.size(0) == 2, "CV should be of size 2 x L x D, where D is the dimension of the location parameter loc"
        assert CV.size(2) == loc.size(0), "CV should be of size 2 x L x D, where D is the dimension of the location parameter loc"
        super(AVFMultivariateNormal, self).__init__(loc, scale_tril=scale_tril)
        self.loc = loc
        self.scale_tril = scale_tril
        self.CV = CV

    def rsample(self, sample_shape=torch.Size()):
        return _AVFMVNSample.apply(self.loc, self.scale_tril, self.CV, sample_shape + self.loc.shape)


class _AVFMVNSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_tril, CV, shape):
        white = loc.new(shape).normal_()
        z = torch.matmul(white, scale_tril.t())
        ctx.save_for_backward(scale_tril, CV, white)
        return loc + z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        L, CV, epsilon = ctx.saved_tensors
        B, C = CV
        g = grad_output
        loc_grad = sum_leftmost(grad_output, -1)

        # compute the rep trick gradient
        epsilon_jb = epsilon.unsqueeze(-2)
        g_ja = g.unsqueeze(-1)
        diff_L_ab = sum_leftmost(g_ja * epsilon_jb, -2)

        # modulate the velocity fields with infinitesimal rotations, i.e. apply the control variate
        gL = torch.matmul(g, L)
        eps_gL_ab = sum_leftmost(gL.unsqueeze(-1) * epsilon.unsqueeze(-2), -2)
        xi_ab = eps_gL_ab - eps_gL_ab.t()
        BC_lab = B.unsqueeze(-1) * C.unsqueeze(-2)
        diff_L_ab += (xi_ab.unsqueeze(0) * BC_lab).sum(0)
        L_grad = torch.tril(diff_L_ab)

        # compute CV grads
        diff_B = (L_grad.unsqueeze(0) * C.unsqueeze(-2) * xi_ab.unsqueeze(0)).sum(2)
        diff_C = (L_grad.t().unsqueeze(0) * B.unsqueeze(-2) * xi_ab.t().unsqueeze(0)).sum(2)
        diff_CV = torch.stack([diff_B, diff_C])

        return loc_grad, L_grad, diff_CV, None
