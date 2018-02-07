from __future__ import absolute_import, division, print_function

from torch.autograd import Variable
from torch.distributions import constraints, transform_to
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat


class SparseVariationalGP(nn.Module):
    """
    Sparse Variational Gaussian Process module.

    :param torch.autograd.Variable X: A tensor of inputs.
    :param torch.autograd.Variable y: A tensor of outputs for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param pyro.contrib.gp.InducingPoints Xu: An inducing-point module for spare approximation.
    :param dict kernel_prior: A mapping from kernel parameter's names to priors.
    :param dict Xu_prior: A mapping from inducing point parameter named 'Xu' to a prior.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, likelihood, Xu, kernel_prior=None,
                 Xu_prior=None, jitter=1e-6):
        super(GPRegression, self).__init__()
        self.X = X
        self.y = y
        self.kernel = kernel
        self.likelihood = likelihood
        self.Xu = Xu

        self.num_data = self.X.size(0)
        self.num_inducing = self.Xu().size(0)

        self.kernel_prior = kernel_prior if kernel_prior is not None else {}
        self.Xu_prior = Xu_prior if Xu_prior is not None else {}

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
        
        Xu_guide_prior = {}
        for p in self.Xu_prior:
            p_MAP_name = pyro.param_with_module_name(self.Xu.name, p) + "_MAP"
            # init params by their prior means
            p_MAP = pyro.param(p_MAP_name, Variable(self.Xu_prior[p].torch_dist.mean.data.clone(),
                                                    requires_grad=True))
            Xu_guide_prior[p] = dist.Delta(p_MAP)
            
        Xu_fn = pyro.random_module(self.Xu.name, self.Xu, Xu_guide_prior)
        Xu = Xu_fn().inducing_points
        # util here

        # construct variational guide
        mu_0 = Variable(Xu.data.new(self.num_inducing).zero_(), requires_grad=True)
        mu = pyro.param("q_u_loc", mu_0)
        unconstrained_Lu_0 = Variable(Xu.data.new(self.num_inducing, self.num_inducing).zero_(),
                                      requires_grad=True)
        unconstrained_Lu = pyro.param("unconstrained_q_u_tril", Lu_0)
        Lu = transform_to(constraints.lower_cholesky)(unconstrained_Lu)
        
        q_f_loc, q_f_cov = self._predict_f(self.X, kernel, Xu, mu, Lu, full_cov=False)

        f = pyro.sample("f", dist.Normal(q_f_loc, q_f_cov))
        return kernel, Xu, mu, Lu

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

        kernel, Xu, mu, Lu = self.guide()
        f_loc, f_cov = _predict_f(Xnew, kernel, Xu, mu, Lu, full_cov=full_cov)
        
        # TODO: use likelihood module to return a stochastic function for y
        # it is better to sample y together with sample f in that function

        return f_loc, f_cov
    
    def _predict_f(X, kernel, Xu, mu, Lu, full_cov=False):
        # Ref: "Scalable Variational Gaussian Process Classification"
        #     by Hensman, J., Matthews, A. G. D. G., & Ghahramani, Z. (2015).
        #
        # u ~ N(mu, Lu @ Lu.T)
        # W := inv(Luu) @ Kuf; V := inv(Luu) @ Lu
        # f_loc = Kfu @ inv(Kuu) @ m = W.T @ inv(Luu) @ m
        # f_cov = Kff - Kfu @ inv(Kuu) @ Kuf + Kfu @ inv(Kuu) @ S @ inv(Kuu) @ Kuf
        #       = Kff - W.T @ W + W.T @ V @ V.T @ W
        #       =: Kff - Qff + K
        
        Kuu = kernel(Xu) + self.jitter.expand(self.num_inducing)
        Kuf = kernel(Xu, X)
        Luu = Kuu.potrf(upper=False)
        
        # combine all tril-solvers to one place
        pack = torch.cat((m.unsqueeze(1), Kuf, Lu), dim=1)
        Luuinv_pack = matrix_triangular_solve_compat(pack, Luu, upper=False)
        Luuinv_u = Luuinv_pack[:, 0].squeeze(1)
        W = Luuinv_pack[:, 1:self.num_data+1]
        V = Luuinv_pack[:, -self.num_inducing:]
        Vt_W = V.t().matmul(W)
        
        f_loc = Luuinv_Kuf.t().matmul(Luuinv_mu)
        
        if full_cov:
            Kff = kernel(X)
            # Qff = Kfu @ inv(Kuu) @ Kuf = W.T @ W
            Qff = W.t().matmul(W)
            # K = Kfu @ inv(Kuu) @ S @ inv(Kuu) @ Kuf = W.T @ V @ V.T @ W
            K = Vt_W.t().matmul(Vt_W)
            f_cov = Kff - Qff + K
        else:
            Kffdiag = kernel(X, diag=True)
            Qffdiag = (W ** 2).sum(dim=0)
            Kdiag = (Vt_W ** 2).sum(dim=0)
            f_cov = Kffdiag - Qffdiag + Kdiag
        
        return f_loc, f_cov
