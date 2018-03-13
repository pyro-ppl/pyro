from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat

from .model import Model


class GPRegression(Model):
    """
    Gaussian Process Regression module.

    References

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor X: A 1D or 2D tensor of input data for training.
    :param torch.Tensor y: A 1D or 2D tensor of output data for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor noise: An optional noise parameter.
    """
    def __init__(self, X, y, kernel, noise=None):
        super(GPRegression, self).__init__()
        self.set_data(X, y)
        self.kernel = kernel

        if noise is None:
            noise = self.X.data.new([1])
        self.noise = Parameter(noise)
        self.set_constraint("noise", constraints.positive)

    def model(self):
        self.set_mode("model")

        kernel = self.kernel
        noise = self.get_param("noise")

        K = kernel(self.X) + noise.expand(self.X.size(0)).diag()
        zero_loc = K.new([0]).expand(K.size(0))
        # correct event_shape for y
        y = self.y.t() if self.y.dim() == 2 else self.y
        pyro.sample("y", dist.MultivariateNormal(zero_loc, K), obs=y)

    def guide(self):
        self.set_mode("guide")

        kernel = self.kernel
        noise = self.get_param("noise")

        return kernel, noise

    def forward(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the parameters of :math:`p(y^*|Xnew) \sim N(\text{loc}, \text{cov})`
        w.r.t. the new input :math:`Xnew`. In case output data is a 2D tensor of shape
        :math:`N \times D`, :math:`loc` is also a 2D tensor of shape :math:`N \times D`.
        Covariance matrix :math:`cov` is always a 2D tensor of shape :math:`N \times N`.

        :param torch.Tensor Xnew: A 1D or 2D tensor.
        :param bool full_cov: Predicts full covariance matrix or just its diagonal.
        :param bool noiseless: Includes noise in the prediction or not.
        :return: loc and covariance matrix of :math:`p(y^*|Xnew)`.
        :rtype: torch.Tensor and torch.Tensor
        """
        self._check_Xnew_shape(Xnew, self.X)

        kernel, noise = self.guide()

        Kff = kernel(self.X)
        Kff = Kff + noise.expand(Kff.size(0)).diag()
        Kfs = kernel(self.X, Xnew)
        Lff = Kff.potrf(upper=False)

        y = self.y.unsqueeze(1) if self.y.dim() == 1 else self.y
        pack = torch.cat((y, Kfs), dim=1)
        Lffinv_pack = matrix_triangular_solve_compat(pack, Lff, upper=False)
        Lffinv_y = Lffinv_pack[:, :y.size(1)].view(self.y.size())
        # W = inv(Lff) @ Kfs
        W = Lffinv_pack[:, y.size(1):]

        # loc = Kfs.T @ inv(Kff) @ y
        loc = W.t().matmul(Lffinv_y)

        # cov = Kss - Ksf @ inv(Kff) @ Kfs
        if full_cov:
            Kss = kernel(Xnew)
            if not noiseless:
                Kss = Kss + noise.expand(Xnew.size(0)).diag()
            Qss = W.t().matmul(W)
            cov = Kss - Qss
        else:
            Kssdiag = kernel(Xnew, diag=True)
            if not noiseless:
                Kssdiag = Kssdiag + noise.expand(Xnew.size(0))
            Qssdiag = (W ** 2).sum(dim=0)
            cov = Kssdiag - Qssdiag

        return loc, cov
