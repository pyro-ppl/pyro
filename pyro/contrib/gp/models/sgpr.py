from __future__ import absolute_import, division, print_function

import math

from torch.autograd import Variable
import torch.nn as nn

import pyro
import pyro.distributions as dist

from .util import _matrix_triangular_solve_compat


class _SparseMultivariateNormal(dist.MultivariateNormal):
    """Sparse Multivariate Normal distribution.

    Approximates covariance matrix in terms D and W, according to
        covariance_matrix ~ D + W.T @ W.

    :param torch.autograd.Variable loc: Mean.
        Must be in 1 dimensional of size N
    :param torch.autograd.Variable covariance_matrix_D_term: D term.
        Must be in 1 dimensional of size N.
    :param torch.autograd.Variable covariance_matrix_D_term: W term.
        Must be in 2 dimensional of size M x N.
    :param float trace_term: A optional term to be added into Mahalabonis term
        according to q(y) = N(y|loc, cov).exp(-1/2 * trace_term).
    """

    def __init__(self, loc, covariance_matrix_D, covariance_matrix_W,
                 trace_term=None, *args, **kwargs):
        covariance_matrix = None
        super(_SparseMultivariateNormal, self).__init__(loc, covariance_matrix, *args, **kwargs)
        self.covariance_matrix_D = covariance_matrix_D
        self.covariance_matrix_W = covariance_matrix_W
        self.trace_term = trace_term if trace_term is not None else 0

    def log_prob(self, value):
        delta = value - self.loc
        logdet, mahalanobis_squared = self._compute_logdet_and_mahalanobis(
            self.covariance_matrix_D, self.covariance_matrix_W, delta, self.trace_term)
        normalization_const = 0.5 * (self.event_shape[-1] * math.log(2 * math.pi) + logdet)
        return -(normalization_const + 0.5 * mahalanobis_squared)

    def _compute_logdet_and_mahalanobis(self, D, W, y, trace_term=0):
        """
        Calculates log determinant and (squared) Mahalanobis term of covariance
        matrix (D + Wt.W), where D is a diagonal matrix, based on the
        "Woodbury matrix identity" and "matrix determinant lemma":
            inv(D + Wt.W) = inv(D) - inv(D).Wt.inv(I + W.inv(D).Wt).W.inv(D)
            log|D + Wt.W| = log|Id + Wt.inv(D).W| + log|D|
        """
        W_Dinv = W / D
        Id = Variable(W.data.new([1])).expand(W.size(0)).diag()
        K = Id + W_Dinv.matmul(W.t())
        L = K.portf(upper=False)
        W_Dinv_y = W_Dinv.matmul(y)
        Linv_W_Dinv_y = _matrix_triangular_solve_compat(W_Dinv_y, L, upper=False)

        logdet = 2 * L.diag().log().sum() + D.diag().log().sum()

        mahalanobis1 = 0.5 * (y * y / D).sum(-1)
        mahalanobis2 = 0.5 * (Linv_W_Dinv_y * Linv_W_Dinv_y).sum()
        mahalanobis_squared = mahalanobis1 - mahalanobis2 + trace_term

        return logdet, mahalanobis_squared


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
        W = _matrix_triangular_solve_compat(Kuf, Luu, upper=False)

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
        pyro.sample("y", _SparseMultivariateNormal(zero_loc, D, W), obs=self.y)

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

        W = _matrix_triangular_solve_compat(Kuf, Luu, upper=False)
        D = self.noise.expand(self.num_data)
        if self.approx == "FITC":
            Kffdiag = kernel(self.X, diag=True)
            Qffdiag = (W ** 2).sum(dim=0)
            D += Kffdiag - Qffdiag

        W_Dinv = W / D
        Id = Variable(W.data.new([1])).expand(self.num_inducing).diag()
        K = Id + W_Dinv.matmul(W.t())
        L = K.potrf(upper=False)

        Ws = _matrix_triangular_solve_compat(Kus, Luu, upper=False)
        Linv_Ws = _matrix_triangular_solve_compat(Ws, L, upper=False)

        # loc = Linv_Ws.T @ inv(L) @ W_Dinv @ y
        W_Dinv_y = W_Dinv.matmul(self.y)
        Linv_W_Dinv_y = _matrix_triangular_solve_compat(W_Dinv_y, L, upper=False)
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
