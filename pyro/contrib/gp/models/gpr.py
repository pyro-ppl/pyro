from __future__ import absolute_import, division, print_function

from torch.autograd import Variable
import torch.nn as nn

import pyro
import pyro.distributions as dist


class GPRegression(nn.Module):
    """
    Gaussian Process regression module.

    :param torch.autograd.Variable X: A tensor of inputs.
    :param torch.autograd.Variable y: A tensor of output data for training.
    :param pyro.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor noise: An optional noise tensor.
    :param dict priors: A mapping from kernel parameter's names to priors.
    """
    def __init__(self, X, y, kernel, noise=None, priors=None):
        super(GPRegression, self).__init__()
        self.X = X
        self.y = y
        self.input_dim = X.size(0)
        self.kernel = kernel
        # TODO: define noise as a Likelihood (another nn.Module beside kernel),
        # then we can train/set prior to it
        if noise is None:
            self.noise = Variable(X.data.new([1]))
        else:
            self.noise = Variable(noise)
        self.priors = priors
        if priors is None:
            self.priors = {}

    def model(self):
        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, self.priors)
        kernel = kernel_fn()
        K = kernel(self.X) + self.noise.repeat(self.input_dim).diag()
        zero_loc = Variable(K.data.new([0]).expand(self.input_dim))
        pyro.sample("f", dist.MultivariateNormal(zero_loc, K), obs=self.y)

    def guide(self):
        guide_priors = {}
        for p in self.priors:
            p_MAP_name = pyro.param_with_module_name(self.kernel.name, p) + "_MAP"
            # init params by their prior means
            p_MAP = pyro.param(p_MAP_name, Variable(self.priors[p].analytic_mean().data.clone(),
                                                    requires_grad=True))
            guide_priors[p] = dist.Delta(p_MAP)
        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, guide_priors)
        return kernel_fn()

    def forward(self, Z):
        """
        Compute the parameters of `p(y|Z) ~ N(loc, covariance_matrix)`
            w.r.t. the new input Z.

        :param torch.autograd.Variable Z: A 2D tensor.
        :return: loc and covariance matrix of p(y|Z).
        :rtype: torch.autograd.Variable and torch.autograd.Variable
        """
        if Z.dim() == 2 and self.X.size(1) != Z.size(1):
            assert ValueError("Train data and test data should have the same feature sizes.")
        if Z.dim() == 1:
            Z = Z.unsqueeze(1)
        kernel = self.guide()
        K = kernel(self.X) + self.noise.repeat(self.input_dim).diag()

        K_xz = kernel(self.X, Z)
        K_zx = K_xz.t()
        K_zz = kernel(Z)

        # TODO Use torch.trtrs or torch.potrs when it supports cuda tensors
        # and is differentiable.
        # Refer to Gaussian processes for machine learning main gpr algorithm
        # L = torch.potrf(K, upper=False)
        # alpha = torch.potrs(L, self.y, upper=False)

        K_inv = K.inverse()
        alpha = K_inv.matmul(self.y)
        loc = K_zx.matmul(alpha)
        covariance_matrix = K_zz - K_zx.matmul(K_inv.matmul(K_xz))

        return loc, covariance_matrix
