# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints

from pyro.distributions.torch import MultivariateNormal
from pyro.distributions.util import sum_leftmost


class AVFMultivariateNormal(MultivariateNormal):
    """Multivariate normal (Gaussian) distribution with transport equation inspired control
    variates (adaptive velocity fields).

    A distribution over vectors in which all the elements have a joint Gaussian density.

    :param torch.Tensor loc: D-dimensional mean vector.
    :param torch.Tensor scale_tril: Cholesky of Covariance matrix; D x D matrix.
    :param torch.Tensor control_var: 2 x L x D tensor that parameterizes the control variate;
        L is an arbitrary positive integer.  This parameter needs to be learned (i.e. adapted) to
        achieve lower variance gradients. In a typical use case this parameter will be adapted
        concurrently with the `loc` and `scale_tril` that define the distribution.


    Example usage::

        control_var = torch.tensor(0.1 * torch.ones(2, 1, D), requires_grad=True)
        opt_cv = torch.optim.Adam([control_var], lr=0.1, betas=(0.5, 0.999))

        for _ in range(1000):
            d = AVFMultivariateNormal(loc, scale_tril, control_var)
            z = d.rsample()
            cost = torch.pow(z, 2.0).sum()
            cost.backward()
            opt_cv.step()
            opt_cv.zero_grad()

    """
    arg_constraints = {"loc": constraints.real, "scale_tril": constraints.lower_triangular,
                       "control_var": constraints.real}

    def __init__(self, loc, scale_tril, control_var):
        if loc.dim() != 1:
            raise ValueError("AVFMultivariateNormal loc must be 1-dimensional")
        if scale_tril.dim() != 2:
            raise ValueError("AVFMultivariateNormal scale_tril must be 2-dimensional")
        if control_var.dim() != 3 or control_var.size(0) != 2 or control_var.size(2) != loc.size(0):
            raise ValueError("control_var should be of size 2 x L x D, where D is the dimension of the location parameter loc")  # noqa: E501
        self.control_var = control_var
        super().__init__(loc, scale_tril=scale_tril)

    def rsample(self, sample_shape=torch.Size()):
        return _AVFMVNSample.apply(self.loc, self.scale_tril, self.control_var, sample_shape + self.loc.shape)


class _AVFMVNSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_tril, control_var, shape):
        white = torch.randn(shape, dtype=loc.dtype, device=loc.device)
        z = torch.matmul(white, scale_tril.t())
        ctx.save_for_backward(scale_tril, control_var, white)
        return loc + z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        L, control_var, epsilon = ctx.saved_tensors
        B, C = control_var
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

        # compute control_var grads
        diff_B = (L_grad.unsqueeze(0) * C.unsqueeze(-2) * xi_ab.unsqueeze(0)).sum(2)
        diff_C = (L_grad.t().unsqueeze(0) * B.unsqueeze(-2) * xi_ab.t().unsqueeze(0)).sum(2)
        diff_CV = torch.stack([diff_B, diff_C])

        return loc_grad, L_grad, diff_CV, None
