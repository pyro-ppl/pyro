from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat

from .model import GPModel


class SparseGPRegression(GPModel):
    """
    Sparse Gaussian Process Regression module.

    This module implements three approximation methods:
    Deterministic Training Conditional (DTC),
    Fully Independent Training Conditional (FITC),
    Variational Free Energy (VFE).

    References

    [1] `A Unifying View of Sparse Approximate Gaussian Process Regression`,
    Joaquin Quinonero-Candela, Carl E. Rasmussen

    [2] `Variational learning of inducing variables in sparse Gaussian processes`,
    Michalis Titsias

    :param torch.Tensor X: A 1D or 2D tensor of input data for training.
    :param torch.Tensor y: A tensor of output data for training with
        ``y.shape[-1]`` equals to number of data points.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor Xu: Initial values for inducing points, which are parameters
        of our model.
    :param torch.Tensor noise: An optional noise tensor.
    :param str approx: One of approximation methods: "DTC", "FITC", and "VFE" (default).
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, Xu, noise=None, approx=None,
                 jitter=1e-6, name="SGPR"):
        super(SparseGPRegression, self).__init__(X, y, kernel, jitter, name)

        noise = self.X.new_ones(()) if noise is None else noise
        self.noise = Parameter(noise)
        self.set_constraint("noise", constraints.greater_than(self.jitter))

        self.Xu = Parameter(Xu)

        if approx is None:
            self.approx = "VFE"
        elif approx in ["DTC", "FITC", "VFE"]:
            self.approx = approx
        else:
            raise ValueError("The sparse approximation method should be one of "
                             "'DTC', 'FITC', 'VFE'.")

    def model(self):
        self.set_mode("model")

        noise = self.get_param("noise")
        Xu = self.get_param("Xu")

        # W = inv(Luu) @ Kuf
        # Qff = Kfu @ inv(Kuu) @ Kuf = W.T @ W
        # Fomulas for each approximation method are
        # DTC:  y_cov = Qff + noise,                   trace_term = 0
        # FITC: y_cov = Qff + diag(Kff - Qff) + noise, trace_term = 0
        # VFE:  y_cov = Qff + noise,                   trace_term = tr(Kff - Qff) / noise
        # y_cov = W.T @ W + D
        # trace_term is added into log_prob

        M = Xu.shape[0]
        Kuu = self.kernel(Xu) + torch.eye(M, out=Xu.new(M, M)) * self.jitter
        Luu = Kuu.potrf(upper=False)
        Kuf = self.kernel(Xu, self.X)
        W = matrix_triangular_solve_compat(Kuf, Luu, upper=False)

        D = noise.expand(W.shape[1])
        trace_term = 0
        if self.approx == "FITC" or self.approx == "VFE":
            Kffdiag = self.kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=0)
            if self.approx == "FITC":
                D = D + Kffdiag - Qffdiag
            else:  # approx = "VFE"
                trace_term += (Kffdiag - Qffdiag).sum() / noise

        zero_loc = self.X.new_zeros(self.X.shape[0])
        if self.y is None:
            f_var = D + W.pow(2).sum(dim=0)
            return zero_loc, f_var
        else:
            y_name = pyro.param_with_module_name(self.name, "y")
            return pyro.sample(y_name,
                               dist.SparseMultivariateNormal(zero_loc, W, D, trace_term)
                                   .reshape(sample_shape=self.y.shape[:-1],
                                            extra_event_dims=self.y.dim()-1),
                               obs=self.y)

    def guide(self):
        self.set_mode("guide")

        noise = self.get_param("noise")
        Xu = self.get_param("Xu")

        return self.kernel, noise, Xu

    def forward(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the parameters of :math:`p(y^*|Xnew) \sim N(\text{loc}, \text{cov})`
        w.r.t. the new input :math:`Xnew`. In case output data is a 2D tensor of shape
        :math:`N \times D`, :math:`loc` is also a 2D tensor of shape :math:`N \times D`.
        Covariance matrix :math:`cov` is always a 2D tensor of shape :math:`N \times N`.

        :param torch.Tensor Xnew: A 1D or 2D tensor.
        :param bool full_cov: Predicts full covariance matrix or just its diagonal.
        :param bool noiseless: Includes noise in the prediction or not.
        :returns: loc and covariance matrix of :math:`p(y^*|Xnew)`.
        :rtype: torch.Tensor and torch.Tensor
        """
        self._check_Xnew_shape(Xnew)
        kernel, noise, Xu = self.guide()

        # W = inv(Luu) @ Kuf
        # Ws = inv(Luu) @ Kus
        # D as in self.model()
        # K = I + W @ inv(D) @ W.T = L @ L.T
        # S = inv[Kuu + Kuf @ inv(D) @ Kfu]
        #   = inv(Luu).T @ inv[I + inv(Luu) @ Kuf @ inv(D) @ Kfu @ inv(Luu).T] @ inv(Luu)
        #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
        #   = inv(Luu).T @ inv(K) @ inv(Luu)
        #   = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
        # loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
        # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
        #     = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

        N = self.X.shape[0]
        M = Xu.shape[0]

        Kuu = kernel(Xu) + torch.eye(M, out=Xu.new(M, M)) * self.jitter
        Luu = Kuu.potrf(upper=False)
        Kus = kernel(Xu, Xnew)
        Kuf = kernel(Xu, self.X)

        W = matrix_triangular_solve_compat(Kuf, Luu, upper=False)
        Ws = matrix_triangular_solve_compat(Kus, Luu, upper=False)
        D = noise.expand(N)
        if self.approx == "FITC":
            Kffdiag = kernel(self.X, diag=True)
            Qffdiag = W.pow(2).sum(dim=0)
            D = D + Kffdiag - Qffdiag

        W_Dinv = W / D
        Id = torch.eye(M, M, out=W.new(M, M))
        K = Id + W_Dinv.matmul(W.t())
        L = K.potrf(upper=False)

        # convert y into 2D tensor for packing
        y_2D = self.y.reshape(-1, N).t()
        W_Dinv_y = W_Dinv.matmul(y_2D)
        pack = torch.cat((W_Dinv_y, Ws), dim=1)
        Linv_pack = matrix_triangular_solve_compat(pack, L, upper=False)
        # unpack
        Linv_W_Dinv_y = Linv_pack[:, :W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1]:]

        loc_shape = self.y.shape[:-1] + (Xnew.shape[0],)
        loc = Linv_W_Dinv_y.t().matmul(Linv_Ws).reshape(loc_shape)

        if full_cov:
            Kss = kernel(Xnew)
            if not noiseless:
                Kss = Kss + noise.expand(Xnew.shape[0]).diag()
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
        else:
            Kssdiag = kernel(Xnew, diag=True)
            if not noiseless:
                Kssdiag = Kssdiag + noise.expand(Xnew.shape[0])
            Qssdiag = Ws.pow(2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=0)

        cov_shape = self.y.shape[:-1] + (Xnew.shape[0], Xnew.shape[0])
        cov = cov.expand(cov_shape)

        return loc, cov
