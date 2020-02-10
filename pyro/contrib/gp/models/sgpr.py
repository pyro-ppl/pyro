# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.nn.module import PyroParam, pyro_method


class SparseGPRegression(GPModel):
    u"""
    Sparse Gaussian Process Regression model.

    In :class:`.GPRegression` model, when the number of input data :math:`X` is large,
    the covariance matrix :math:`k(X, X)` will require a lot of computational steps to
    compute its inverse (for log likelihood and for prediction). By introducing an
    additional inducing-input parameter :math:`X_u`, we can reduce computational cost
    by approximate :math:`k(X, X)` by a low-rank Nymstr\u00F6m approximation :math:`Q`
    (see reference [1]), where

    .. math:: Q = k(X, X_u) k(X,X)^{-1} k(X_u, X).

    Given inputs :math:`X`, their noisy observations :math:`y`, and the inducing-input
    parameters :math:`X_u`, the model takes the form:

    .. math::
        u & \\sim \\mathcal{GP}(0, k(X_u, X_u)),\\\\
        f & \\sim q(f \\mid X, X_u) = \\mathbb{E}_{p(u)}q(f\\mid X, X_u, u),\\\\
        y & \\sim f + \\epsilon,

    where :math:`\\epsilon` is Gaussian noise and the conditional distribution
    :math:`q(f\\mid X, X_u, u)` is an approximation of

    .. math:: p(f\\mid X, X_u, u) = \\mathcal{N}(m, k(X, X) - Q),

    whose terms :math:`m` and :math:`k(X, X) - Q` is derived from the joint
    multivariate normal distribution:

    .. math:: [f, u] \\sim \\mathcal{GP}(0, k([X, X_u], [X, X_u])).

    This class implements three approximation methods:

    + Deterministic Training Conditional (DTC):

        .. math:: q(f\\mid X, X_u, u) = \\mathcal{N}(m, 0),

      which in turns will imply

        .. math:: f \\sim \\mathcal{N}(0, Q).

    + Fully Independent Training Conditional (FITC):

        .. math:: q(f\\mid X, X_u, u) = \\mathcal{N}(m, diag(k(X, X) - Q)),

      which in turns will correct the diagonal part of the approximation in DTC:

        .. math:: f \\sim \\mathcal{N}(0, Q + diag(k(X, X) - Q)).

    + Variational Free Energy (VFE), which is similar to DTC but has an additional
      `trace_term` in the model's log likelihood. This additional term makes "VFE"
      equivalent to the variational approach in :class:`.SparseVariationalGP`
      (see reference [2]).

    .. note:: This model has :math:`\\mathcal{O}(NM^2)` complexity for training,
        :math:`\\mathcal{O}(NM^2)` complexity for testing. Here, :math:`N` is the number
        of train inputs, :math:`M` is the number of inducing inputs.

    References:

    [1] `A Unifying View of Sparse Approximate Gaussian Process Regression`,
    Joaquin Qui\u00F1onero-Candela, Carl E. Rasmussen

    [2] `Variational learning of inducing variables in sparse Gaussian processes`,
    Michalis Titsias

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor Xu: Initial values for inducing points, which are parameters
        of our model.
    :param torch.Tensor noise: Variance of Gaussian noise of this model.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param str approx: One of approximation methods: "DTC", "FITC", and "VFE"
        (default).
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :param str name: Name of this model.
    """
    def __init__(self, X, y, kernel, Xu, noise=None, mean_function=None, approx=None, jitter=1e-6):
        super().__init__(X, y, kernel, mean_function, jitter)

        self.Xu = Parameter(Xu)

        noise = self.X.new_tensor(1.) if noise is None else noise
        self.noise = PyroParam(noise, constraints.positive)

        if approx is None:
            self.approx = "VFE"
        elif approx in ["DTC", "FITC", "VFE"]:
            self.approx = approx
        else:
            raise ValueError("The sparse approximation method should be one of "
                             "'DTC', 'FITC', 'VFE'.")

    @pyro_method
    def model(self):
        self.set_mode("model")

        # W = (inv(Luu) @ Kuf).T
        # Qff = Kfu @ inv(Kuu) @ Kuf = W @ W.T
        # Fomulas for each approximation method are
        # DTC:  y_cov = Qff + noise,                   trace_term = 0
        # FITC: y_cov = Qff + diag(Kff - Qff) + noise, trace_term = 0
        # VFE:  y_cov = Qff + noise,                   trace_term = tr(Kff-Qff) / noise
        # y_cov = W @ W.T + D
        # trace_term is added into log_prob

        N = self.X.size(0)
        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
        Luu = Kuu.cholesky()
        Kuf = self.kernel(self.Xu, self.X)
        W = Kuf.triangular_solve(Luu, upper=False)[0].t()

        D = self.noise.expand(N)
        if self.approx == "FITC" or self.approx == "VFE":
            Kffdiag = self.kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=-1)
            if self.approx == "FITC":
                D = D + Kffdiag - Qffdiag
            else:  # approx = "VFE"
                trace_term = (Kffdiag - Qffdiag).sum() / self.noise
                trace_term = trace_term.clamp(min=0)

        zero_loc = self.X.new_zeros(N)
        f_loc = zero_loc + self.mean_function(self.X)
        if self.y is None:
            f_var = D + W.pow(2).sum(dim=-1)
            return f_loc, f_var
        else:
            if self.approx == "VFE":
                pyro.factor(self._pyro_get_fullname("trace_term"), -trace_term / 2.)

            return pyro.sample(self._pyro_get_fullname("y"),
                               dist.LowRankMultivariateNormal(f_loc, W, D)
                                   .expand_by(self.y.shape[:-1])
                                   .to_event(self.y.dim() - 1),
                               obs=self.y)

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

    def forward(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`), the inducing-point
            parameter ``Xu``, together with kernel's parameters have been learned from
            a training procedure (MCMC or SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        # W = inv(Luu) @ Kuf
        # Ws = inv(Luu) @ Kus
        # D as in self.model()
        # K = I + W @ inv(D) @ W.T = L @ L.T
        # S = inv[Kuu + Kuf @ inv(D) @ Kfu]
        #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
        #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
        #   = inv(Luu).T @ inv(K) @ inv(Luu)
        #   = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
        # loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
        # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
        #     = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

        N = self.X.size(0)
        M = self.Xu.size(0)

        # TODO: cache these calculations to get faster inference

        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
        Luu = Kuu.cholesky()

        Kuf = self.kernel(self.Xu, self.X)

        W = Kuf.triangular_solve(Luu, upper=False)[0]
        D = self.noise.expand(N)
        if self.approx == "FITC":
            Kffdiag = self.kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=0)
            D = D + Kffdiag - Qffdiag

        W_Dinv = W / D
        K = W_Dinv.matmul(W.t()).contiguous()
        K.view(-1)[::M + 1] += 1  # add identity matrix to K
        L = K.cholesky()

        # get y_residual and convert it into 2D tensor for packing
        y_residual = self.y - self.mean_function(self.X)
        y_2D = y_residual.reshape(-1, N).t()
        W_Dinv_y = W_Dinv.matmul(y_2D)

        # End caching ----------

        Kus = self.kernel(self.Xu, Xnew)
        Ws = Kus.triangular_solve(Luu, upper=False)[0]
        pack = torch.cat((W_Dinv_y, Ws), dim=1)
        Linv_pack = pack.triangular_solve(L, upper=False)[0]
        # unpack
        Linv_W_Dinv_y = Linv_pack[:, :W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1]:]

        C = Xnew.size(0)
        loc_shape = self.y.shape[:-1] + (C,)
        loc = Linv_W_Dinv_y.t().matmul(Linv_Ws).reshape(loc_shape)

        if full_cov:
            Kss = self.kernel(Xnew).contiguous()
            if not noiseless:
                Kss.view(-1)[::C + 1] += self.noise  # add noise to the diagonal
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
            cov_shape = self.y.shape[:-1] + (C, C)
            cov = cov.expand(cov_shape)
        else:
            Kssdiag = self.kernel(Xnew, diag=True)
            if not noiseless:
                Kssdiag = Kssdiag + self.noise
            Qssdiag = Ws.pow(2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=0)
            cov_shape = self.y.shape[:-1] + (C,)
            cov = cov.expand(cov_shape)

        return loc + self.mean_function(Xnew), cov
