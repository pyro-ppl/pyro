from __future__ import absolute_import, division, print_function

from torch.autograd import Variable
import torch.nn as nn

import pyro
import pyro.distributions as dist

from .util import _matrix_triangular_solve_compat


class GPRegression(nn.Module):
    """
    Gaussian Process Regression module.

    :param torch.autograd.Variable X: A tensor of inputs.
    :param torch.autograd.Variable y: A tensor of output data for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor noise: An optional noise tensor.
    :param dict priors: A mapping from kernel parameter's names to priors.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, noise=None, kernel_prior=None, jitter=1e-6):
        super(GPRegression, self).__init__()
        self.X = X
        self.y = y
        self.kernel = kernel

        self.num_data = self.X.size(0)

        # TODO: define noise as a Likelihood (a nn.Module)
        self.noise = Variable(noise) if noise is not None else Variable(X.data.new([1]))

        self.kernel_prior = kernel_prior if kernel_prior is not None else {}

        self.jitter = Variable(self.X.data.new([jitter]))

    def model(self):
        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, self.kernel_prior)
        kernel = kernel_fn()

        K = kernel(self.X) + self.noise.expand(self.num_data).diag()
        zero_loc = Variable(K.data.new([0])).expand(self.num_data)
        pyro.sample("y", dist.MultivariateNormal(zero_loc, K), obs=self.y)

    def guide(self):
        kernel_guide_prior = {}
        for p in self.kernel_prior:
            p_MAP_name = pyro.param_with_module_name(self.kernel.name, p) + "_MAP"
            # init params by their prior means
            p_MAP = pyro.param(p_MAP_name, Variable(self.kernel_prior[p].analytic_mean().data.clone(),
                                                    requires_grad=True))
            kernel_guide_prior[p] = dist.Delta(p_MAP)

        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, kernel_guide_prior)
        kernel = kernel_fn()

        return kernel

    def forward(self, Xnew, full_cov=False, noiseless=True):
        """
        Compute the parameters of `p(y|Xnew) ~ N(loc, cov)`
            w.r.t. the new input Xnew.

        :param torch.autograd.Variable Xnew: A 2D tensor.
        :return: loc and covariance matrix of p(y|Xnew).
        :rtype: torch.autograd.Variable and torch.autograd.Variable
        """
        if Xnew.dim() == 2 and self.X.size(1) != Xnew.size(1):
            assert ValueError("Train data and test data should have the same feature sizes.")
        if Xnew.dim() == 1:
            Xnew = Xnew.unsqueeze(1)

        kernel = self.guide()
        Kff = kernel(self.X)
        if noiseless:
            Kff += self.jitter.expand(self.num_data).diag()
        else:
            Kff += self.noise.expand(self.num_data).diag()
        Kfs = kernel(self.X, Xnew)
        Lff = Kff.potrf(upper=False)

        Lffinv_y = _matrix_triangular_solve_compat(self.y, Lff, upper=False)
        Lffinv_Kfs = _matrix_triangular_solve_compat(Kfs, Lff, upper=False)

        # loc = Kfs.T @ inv(Kff) @ y
        loc = Lffinv_Kfs.t().matmul(Lffinv_y)

        # cov = Kss - Ksf @ inv(Kff) @ Kfs
        if full_cov:
            Kss = kernel(Xnew)
            Qss = Lffinv_Kfs.t().matmul(Lffinv_Kfs)
            cov = Kss - Qss
        else:
            Kssdiag = kernel(Xnew, diag=True)
            Qssdiag = (Lffinv_Kfs ** 2).sum(dim=0)
            cov = Kssdiag - Qssdiag

        return loc, cov
