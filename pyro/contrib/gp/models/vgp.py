from __future__ import absolute_import, division, print_function

from torch.autograd import Variable
from torch.distributions import constraints, transform_to
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat


class VariationalGP(nn.Module):
    """
    Variational Gaussian Process module.

    :param torch.autograd.Variable X: A tensor of inputs.
    :param torch.autograd.Variable y: A tensor of outputs for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param dict kernel_prior: A mapping from kernel parameter's names to priors.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, likelihood, kernel_prior=None, jitter=1e-6):
        super(VariationalGP, self).__init__()
        self.X = X
        self.y = y
        self.kernel = kernel
        self.likelihood = likelihood

        self.num_data = self.X.size(0)

        self.kernel_prior = kernel_prior if kernel_prior is not None else {}

        self.jitter = Variable(self.X.data.new([jitter]))

    def model(self):
        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, self.kernel_prior)
        kernel = kernel_fn()

        zero_loc = Variable(self.X.data.new([0])).expand(self.num_data)
        Kffdiag = kernel(self.X, diag=True)
        
        f = pyro.sample("f", dist.Normal(zero_loc, Kffdiag))
        likelihood = pyro.condition(self.likelihood, data={"y": self.y})
        return likelihood(f)
    
    def guide(self):
        # TODO: refactor/remove from here
        kernel_guide_prior = {}
        for p in self.kernel_prior:
            p_MAP_name = pyro.param_with_module_name(self.kernel.name, p) + "_MAP"
            # init params by their prior means
            p_MAP = pyro.param(p_MAP_name, Variable(self.kernel_prior[p].torch_dist.mean.data.clone(),
                                                    requires_grad=True))
            kernel_guide_prior[p] = dist.Delta(p_MAP)

        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, kernel_guide_prior)
        kernel = kernel_fn()
        # util here
        
        # construct variational guide
        mu_0 = Variable(Xu.data.new(self.num_inducing).zero_(), requires_grad=True)
        mu = pyro.param("q_u_loc", mu_0)
        unconstrained_Lu_0 = Variable(Xu.data.new(self.num_inducing, self.num_inducing).zero_(),
                                      requires_grad=True)
        unconstrained_Lu = pyro.param("unconstrained_q_u_tril", Lu_0)
        Lu = transform_to(constraints.lower_cholesky)(unconstrained_Lu)
        #
        return kernel

    def forward(self, Xnew, full_cov=False):
        """
        Compute the parameters of `f* ~ N(f_loc, f_cov)` and a stochastic function
        for `y* ~ self.likelihood(f*)`.

        :param torch.autograd.Variable Xnew: A 2D tensor.
        :param bool full_cov: Predict full covariance matrix of f or just its diagonal.
        :return: loc and covariance matrix of p(y|Xnew).
        :rtype: torch.autograd.Variable and torch.autograd.Variable
        """
        if Xnew.dim() == 2 and self.X.size(1) != Xnew.size(1):
            assert ValueError("Train data and test data should have the same feature sizes.")
        if Xnew.dim() == 1:
            Xnew = Xnew.unsqueeze(1)

        # TODO: implement
