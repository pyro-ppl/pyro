from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat

from .model import Model


class SparseGPRegression(Model):
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
        ``y.size(0)`` equals to number of data points.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor Xu: Initial values for inducing points, which are parameters
        of our model.
    :param torch.Tensor noise: An optional noise tensor.
    :param str approx: One of approximation methods: "DTC", "FITC", and "VFE" (default).
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, Xu, noise=None, approx=None, jitter=1e-6):
        latent_shape = torch.Size([])
        super(SparseGPRegression, self).__init__(X, y, kernel, latent_shape, jitter)

        noise = self.X.new([1]) if noise is None else noise
        self.noise = Parameter(noise)
        self.set_constraint("noise", constraints.greater_than(self.jitter))

        self.Xu = Parameter(Xu)

        if approx is None:
            self.approx = "VFE"
        elif approx in ["DTC", "FITC", "VFE"]:
            self.approx = approx
        else:
            raise ValueError("The sparse approximation method should be one of 'DTC', "
                             "'FITC', 'VFE'.")

    def model(self):
        self.set_mode("model")

        kernel = self.kernel
        noise = self.get_param("noise")
        Xu = self.get_param("Xu")

        Kuu = kernel(Xu) + self.jitter.expand(Xu.size(0)).diag()
        Kuf = kernel(Xu, self.X)
        Luu = Kuu.potrf(upper=False)
        # W = inv(Luu) @ Kuf
        W = matrix_triangular_solve_compat(Kuf, Luu, upper=False)

        D = noise.expand(W.size(1))
        trace_term = 0
        if self.approx == "FITC" or self.approx == "VFE":
            Kffdiag = kernel(self.X, diag=True)
            # Qff = Kfu @ inv(Kuu) @ Kuf = W.T @ W
            Qffdiag = (W ** 2).sum(dim=0)

            if self.approx == "FITC":
                D = D + Kffdiag - Qffdiag
            else:  # approx = "VFE"
                trace_term += (Kffdiag - Qffdiag).sum() / noise

        # DTC: cov = Qff + noise, trace_term = 0
        # FITC: cov = Qff + diag(Kff - Qff) + noise, trace_term = 0
        # VFE: cov = Qff + noise, trace_term = tr(Kff - Qff) / noise
        # convert y_shape from N x D to D x N
        y = self.y.permute(list(range(1, self.y.dim())) + [0])
        zero_loc = self.X.new([0]).expand_as(y)
        pyro.sample("y", dist.SparseMultivariateNormal(zero_loc, D, W, trace_term)
                    .reshape(extra_event_dims=y.dim()-1), obs=y)

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

        Kuu = kernel(Xu) + self.jitter.expand(Xu.size(0)).diag()
        Kus = kernel(Xu, Xnew)
        Kuf = kernel(Xu, self.X)
        Luu = Kuu.potrf(upper=False)

        # loc = Ksu @ S @ Kuf @ inv(D) @ y
        # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
        # S = inv[Kuu + Kuf @ inv(D) @ Kfu]
        #   = inv(Luu).T @ inv[I + inv(Luu) @ Kuf @ inv(D) @ Kfu @ inv(Luu).T] @ inv(Luu)
        #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
        #   = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)

        W = matrix_triangular_solve_compat(Kuf, Luu, upper=False)
        D = noise.expand(W.size(1))
        if self.approx == "FITC":
            Kffdiag = kernel(self.X, diag=True)
            Qffdiag = (W ** 2).sum(dim=0)
            D = D + Kffdiag - Qffdiag

        W_Dinv = W / D
        M = W.size(0)
        Id = torch.eye(M, M, out=W.new(M, M))
        K = Id + W_Dinv.matmul(W.t())
        L = K.potrf(upper=False)

        Ws = matrix_triangular_solve_compat(Kus, Luu, upper=False)

        # loc = Linv_Ws.T @ inv(L) @ W_Dinv @ y
        # convert y into 2D tensor
        y_temp = self.y.view(self.y.size(0), -1)
        W_Dinv_y = W_Dinv.matmul(y_temp)
        pack = torch.cat((W_Dinv_y, Ws), dim=1)
        Linv_pack = matrix_triangular_solve_compat(pack, L, upper=False)
        Linv_W_Dinv_y = Linv_pack[:, :W_Dinv_y.size(1)]
        Linv_Ws = Linv_pack[:, W_Dinv_y.size(1):]

        loc_shape = Xnew.size()[:1] + self.y.size()[1:]
        loc = Linv_Ws.t().matmul(Linv_W_Dinv_y).view(loc_shape)

        # cov = Kss - Ws.T @ Ws + Linv_Ws.T @ Linv_Ws
        if full_cov:
            Kss = kernel(Xnew)
            if not noiseless:
                Kss = Kss + noise.expand(Xnew.size(0)).diag()
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
        else:
            Kssdiag = kernel(Xnew, diag=True)
            if not noiseless:
                Kssdiag = Kssdiag + noise.expand(Xnew.size(0))
            Qssdiag = (Ws ** 2).sum(dim=0)
            cov = Kssdiag - Qssdiag + (Linv_Ws ** 2).sum(dim=0)

        # expand cov from N to N x 1 to N x D or N x N to N x N x 1 to N x N x D
        cov_shape_pre = cov.size() + torch.Size([1] * (self.y.dim()-1))
        cov_shape = cov.size() + self.y.size()[1:]
        cov = cov.view(cov_shape_pre).expand(cov_shape)

        return loc, cov
