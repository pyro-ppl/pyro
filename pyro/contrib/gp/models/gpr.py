from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.util import ng_zeros


class GPRegression(nn.Module):
    """
    Gaussian Process regression module.
    """
    def __init__(self, X, y, kernel, noise=torch.ones(1), priors={}):
        super(GPRegression, self).__init__()
        self.X = X
        self.y = y
        self.input_dim = X.size(0)
        self.kernel = kernel
        # TODO: define noise as a nn.Module, so we can train/set prior to it
        self.noise = Variable(noise.type_as(X.data))
        self.priors = priors
        
    def model(self):
        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, self.priors)
        kernel = kernel_fn()
        K = kernel.K(self.X) + self.noise.repeat(self.input_dim).diag()
        zero_loc = Variable(torch.zeros(self.input_dim).type_as(K))
        pyro.sample("f", dist.MultivariateNormal(ng_zeros(self.input_dim), K), obs=self.y)
        
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
        K = kernel.K(self.X) + self.noise.repeat(self.input_dim).diag()
        K_xz = kernel(self.X, Z)
        K_zx = K_xz.t()
        K_zz = kernel.K(Z)
        loc = K_zx.matmul(self.y.gesv(K)[0]).squeeze(1)
        covariance_matrix = K_zz - K_zx.matmul(K_xz.gesv(K)[0])
        return loc, covariance_matrix
