from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable
from torch.distributions import constraints, transform_to
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat


class SparseVariationalGP(nn.Module):
    """
    Sparse Variational Gaussian Process module.

    References

    [1] `Scalable variational Gaussian process classification`
    James Hensman, Alexander G. de G. Matthews, Zoubin Ghahramani

    :param torch.autograd.Variable X: A tensor of inputs.
    :param torch.autograd.Variable y: A tensor of outputs for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param pyro.contrib.gp.InducingPoints Xu: An inducing-point module for spare approximation.
    :param dict kernel_prior: A mapping from kernel parameter's names to priors.
    :param dict Xu_prior: A mapping from inducing point parameter named 'Xu' to a prior.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, likelihood, Xu, kernel_prior=None, Xu_prior=None, jitter=1e-6):
        super(SparseVariationalGP, self).__init__()
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

        Xu_fn = pyro.random_module(self.Xu.name, self.Xu, self.Xu_prior)
        Xu = Xu_fn().inducing_points

        Kuu = kernel(Xu) + self.jitter.expand(self.num_inducing).diag()
        Kuf = kernel(Xu, self.X)
        Luu = Kuu.potrf(upper=False)

        zero_loc = Variable(Xu.data.new([0])).expand(self.num_inducing)
        u = pyro.sample("u", dist.MultivariateNormal(loc=zero_loc, scale_tril=Luu))

        # p(f|u) ~ N(f|mf, Kf)
        # mf = Kfu @ inv(Kuu) @ u; Kf = Kff - Kfu @ inv(Kuu) @ Kuf
        pack = torch.cat((u.unsqueeze(1), Kuf), dim=1)
        Luuinv_pack = matrix_triangular_solve_compat(pack, Luu, upper=False)
        Luuinv_u = Luuinv_pack[:, 0]
        Luuinv_Kuf = Luuinv_pack[:, 1:]

        mf = Luuinv_Kuf.t().matmul(Luuinv_u)
        Kffdiag = kernel(self.X, diag=True)
        Qffdiag = (Luuinv_Kuf ** 2).sum(dim=0)
        Kfdiag = Kffdiag - Qffdiag

        f = dist.Normal(mf, Kfdiag).sample()
        self.likelihood(f, obs=self.y)

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

        # define variational parameters
        mu_0 = Variable(Xu.data.new(self.num_inducing).zero_(), requires_grad=True)
        mu = pyro.param("u_loc", mu_0)
        unconstrained_Lu_0 = Variable(Xu.data.new(self.num_inducing, self.num_inducing).zero_(),
                                      requires_grad=True)
        unconstrained_Lu = pyro.param("unconstrained_u_tril", unconstrained_Lu_0)
        Lu = transform_to(constraints.lower_cholesky)(unconstrained_Lu)

        pyro.sample("u", dist.MultivariateNormal(loc=mu, scale_tril=Lu))
        return kernel, Xu, mu, Lu

    def forward(self, Xnew, full_cov=False):
        """
        Compute the parameters of `f* ~ N(f*_loc, f*_cov)` according to
        `p(f*,u|y) = p(f*|u).p(u|y) ~ p(f*|u).q(u)`, then marginalize out variable `u`.

        :param torch.autograd.Variable Xnew: A 2D tensor.
        :param bool full_cov: Predict full covariance matrix of f or just its diagonal.
        :return: loc and covariance matrix of p(f*|Xnew).
        :rtype: torch.autograd.Variable and torch.autograd.Variable
        """
        if Xnew.dim() == 2 and self.X.size(1) != Xnew.size(1):
            assert ValueError("Train data and test data should have the same feature sizes.")
        if Xnew.dim() == 1:
            Xnew = Xnew.unsqueeze(1)

        kernel, Xu, mu, Lu = self.guide()

        # W := inv(Luu) @ Kus; V := inv(Luu) @ Lu
        # f_loc = Ksu @ inv(Kuu) @ mu = W.T @ inv(Luu) @ mu
        # f_cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ inv(Kuu) @ S @ inv(Kuu) @ Kus
        #       = Kss - W.T @ W + W.T @ V @ V.T @ W
        #       =: Kss - Qss + K

        Kuu = kernel(Xu) + self.jitter.expand(self.num_inducing)
        Kus = kernel(Xu, Xnew)
        Luu = Kuu.potrf(upper=False)

        # combine all tril-solvers to one place
        pack = torch.cat((mu.unsqueeze(1), Kus, Lu), dim=1)
        Luuinv_pack = matrix_triangular_solve_compat(pack, Luu, upper=False)
        Luuinv_mu = Luuinv_pack[:, 0]
        W = Luuinv_pack[:, 1:-self.num_inducing]
        V = Luuinv_pack[:, -self.num_inducing:]
        Vt_W = V.t().matmul(W)

        fs_loc = W.t().matmul(Luuinv_mu)

        if full_cov:
            Kss = kernel(Xnew)
            # Qss = Ksu @ inv(Kuu) @ Kus = W.T @ W
            Qss = W.t().matmul(W)
            # K = Ksu @ inv(Kuu) @ S @ inv(Kuu) @ Kus = W.T @ V @ V.T @ W
            K = Vt_W.t().matmul(Vt_W)
            fs_cov = Kss - Qss + K
        else:
            Kssdiag = kernel(Xnew, diag=True)
            Qssdiag = (W ** 2).sum(dim=0)
            Kdiag = (Vt_W ** 2).sum(dim=0)
            fs_cov = Kssdiag - Qssdiag + Kdiag

        return fs_loc, fs_cov
