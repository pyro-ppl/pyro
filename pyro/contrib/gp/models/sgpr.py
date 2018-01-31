from __future__ import absolute_import, division, print_function

from torch.autograd import Variable
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat


class SparseGPRegression(nn.Module):
    """
    Sparse Gaussian Process Regression module.

    :param torch.autograd.Variable X: A tensor of inputs.
    :param torch.autograd.Variable y: A tensor of outputs for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param pyro.contrib.gp.InducingPoints Xu: An inducing-point module for spare approximation.
    :param torch.Tensor noise: An optional noise tensor.
    :param str approx: One of approximation methods: 'DTC', 'FITC', and 'VFE' (default).
    :param dict kernel_prior: A mapping from kernel parameter's names to priors.
    :param dict Xu_prior: A mapping from inducing point parameter named 'Xu' to a prior.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, Xu, noise=None, approx=None, kernel_prior=None,
                 Xu_prior=None, jitter=1e-6):
        super(SparseGPRegression, self).__init__()
        self.X = X
        self.y = y
        self.kernel = kernel
        self.Xu = Xu

        self.num_data = self.X.size(0)
        self.num_inducing = self.Xu.size(0)

        # TODO: define noise as a Likelihood (a nn.Module)
        self.noise = Variable(noise) if noise is not None else Variable(X.data.new([1]))

        if approx is None:
            self.approx = "VFE"
        elif approx in ["DTC", "FITC", "VFE"]:
            self.approx = approx
        else:
            raise ValueError("The sparse approximation method should be one of 'DTC', "
                             "'FITC', 'VFE'.")

        self.kernel_prior = kernel_prior if kernel_prior is not None else {}
        self.Xu_prior = Xu_prior if Xu_prior is not None else {}

        self.jitter = Variable(self.X.data.new([jitter]))

    def model(self):
        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, self.kernel_priors)
        kernel = kernel_fn()

        Xu_fn = pyro.random_module(self.Xu.name, self.Xu, self.Xu_priors)
        Xu = Xu_fn()()  # Xu is a nn.Module

        Kuu = kernel(Xu) + self.jitter.expand(self.num_inducing)
        Kuf = kernel(Xu, self.X)
        Luu = Kuu.potrf(upper=False)
        # W = inv(Luu) @ Kuf
        W = matrix_triangular_solve_compat(Kuf, Luu, upper=False)

        D = self.noise.expand(self.num_data)
        trace_term = 0
        if self.approx == "FITC" or self.approx == "VFE":
            Kffdiag = kernel(self.X, diag=True)
            # Qff = Kfu @ inv(Kuu) @ Kuf = W.T @ W
            Qffdiag = (W ** 2).sum(dim=0)

            if self.approx == "FITC":
                D += Kffdiag - Qffdiag
            else:  # approx = "VFE"
                trace_term += (Kffdiag - Qffdiag).sum() / self.noise

        zero_loc = Variable(D.data.new([0])).expand(self.num_data)
        # DTC: cov = Qff + noise, trace_term = 0
        # FITC: cov = Qff + diag(Kff - Qff) + noise, trace_term = 0
        # VFE: cov = Qff + noise, trace_term = tr(Kff - Qff) / noise
        pyro.sample("y", dist.SparseMultivariateNormal(zero_loc, D, W), obs=self.y)

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

        Xu_guide_prior = {}
        for p in self.Xu_prior:
            p_MAP_name = pyro.param_with_module_name(self.Xu.name, p) + "_MAP"
            # init params by their prior means
            p_MAP = pyro.param(p_MAP_name, Variable(self.Xu_prior[p].analytic_mean().data.clone(),
                                                    requires_grad=True))
            Xu_guide_prior[p] = dist.Delta(p_MAP)

        Xu_fn = pyro.random_module(self.Xu.name, self.Xu, Xu_guide_prior)
        Xu = Xu_fn()()

        return kernel, Xu

    def forward(self, Xnew, full_cov=False, noiseless=True):
        """
        Computes the parameters of `p(y|Xnew) ~ N(loc, cov)`
        w.r.t. the new input Xnew.

        :param torch.autograd.Variable Xnew: A 2D tensor.
        :param bool full_cov: Predict
        :
        :return: loc and covariance matrix of p(y|Xnew).
        :rtype: torch.autograd.Variable and torch.autograd.Variable
        """
        if Xnew.dim() == 2 and self.X.size(1) != Xnew.size(1):
            assert ValueError("Train data and test data should have the same feature sizes.")
        if Xnew.dim() == 1:
            Xnew = Xnew.unsqueeze(1)

        kernel, Xu = self.guide()
        Kuu = kernel(Xu) + self.jitter.expand(self.num_inducing)
        Kus = kernel(Xu, Xnew)
        Kuf = kernel(Xu, self.X)
        Luu = Kuu.potrf(upper=False)

        # Ref: "A Unifying View of Sparse Approximate Gaussian Process Regression"
        #    by Quiñonero-Candela, J., & Rasmussen, C. E. (2005).
        #
        # loc = Ksu @ S @ Kuf @ inv(D) @ y
        # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
        # S = inv[Kuu + Kuf @ inv(D) @ Kfu]
        #   = inv(Luu).T @ inv[I + inv(Luu) @ Kuf @ inv(D) @ Kfu @ inv(Luu).T] @ inv(Luu)
        #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
        #   = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)

        W = matrix_triangular_solve_compat(Kuf, Luu, upper=False)
        D = self.noise.expand(self.num_data)
        if self.approx == "FITC":
            Kffdiag = kernel(self.X, diag=True)
            Qffdiag = (W ** 2).sum(dim=0)
            D += Kffdiag - Qffdiag

        W_Dinv = W / D
        Id = Variable(W.data.new([1])).expand(self.num_inducing).diag()
        K = Id + W_Dinv.matmul(W.t())
        L = K.potrf(upper=False)

        Ws = matrix_triangular_solve_compat(Kus, Luu, upper=False)
        Linv_Ws = matrix_triangular_solve_compat(Ws, L, upper=False)

        # loc = Linv_Ws.T @ inv(L) @ W_Dinv @ y
        W_Dinv_y = W_Dinv.matmul(self.y)
        Linv_W_Dinv_y = matrix_triangular_solve_compat(W_Dinv_y, L, upper=False)
        loc = Linv_Ws.t().matmul(Linv_W_Dinv_y)

        # cov = Kss - Ws.T @ Ws + Linv_Ws.T @ Linv_Ws
        if full_cov:
            Kss = kernel(Xnew)
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
        else:
            Kssdiag = kernel(Xnew, diag=True)
            Qssdiag = (Ws ** 2).sum(dim=0)
            cov = Kssdiag - Qssdiag + (Linv_Ws ** 2).sum(dim=0)

        return loc, cov
