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
    :param torch.Tensor y: A tensor of output data for training with
        ``y.size(0)`` equals to number of data points.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor noise: An optional noise parameter.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, noise=None, jitter=1e-6):
        latent_shape = torch.Size([])
        super(GPRegression, self).__init__(X, y, kernel, latent_shape, jitter)

        if noise is None:
            noise = self.X.data.new([1])
        self.noise = Parameter(noise)
        self.set_constraint("noise", constraints.greater_than(self.jitter))

    def model(self):
        self.set_mode("model")

        kernel = self.kernel
        noise = self.get_param("noise")

        zero_loc = self.X.new([0]).expand(self.X.size(0))
        K = kernel(self.X) + noise.expand(self.X.size(0)).diag()
        # convert y_shape from N x D to D x N
        y = self.y.permute(*range(self.y.dim())[1:], 0)
        pyro.sample("y", dist.MultivariateNormal(zero_loc, K)
                    .reshape(sample_shape=y.size()[:-1], extra_event_dims=y.dim()-1),
                    obs=y)

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
        :returns: loc and covariance matrix of :math:`p(y^*|Xnew)`.
        :rtype: torch.Tensor and torch.Tensor
        """
        self._check_Xnew_shape(Xnew, self.X)

        kernel, noise = self.guide()

        Kff = kernel(self.X)
        Kff = Kff + noise.expand(Kff.size(0)).diag()
        Kfs = kernel(self.X, Xnew)
        Lff = Kff.potrf(upper=False)

        # convert y into 2D tensor before packing
        y_temp = self.y.view(self.y.size(0), -1)
        pack = torch.cat((y_temp, Kfs), dim=1)
        Lffinv_pack = matrix_triangular_solve_compat(pack, Lff, upper=False)
        # unpack
        Lffinv_y = Lffinv_pack[:, :y_temp.size(1)]
        # W = inv(Lff) @ Kfs
        W = Lffinv_pack[:, y_temp.size(1):]

        # loc = Kfs.T @ inv(Kff) @ y
        loc_shape = Xnew.size()[:1] + self.y.size()[1:]
        loc = W.t().matmul(Lffinv_y).view(loc_shape)

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

        # expand cov from N to N x 1 to N x D or N x N to N x N x 1 to N x N x D
        cov_shape_pre = cov.size() + torch.Size([1] * (self.y.dim()-1))
        cov_shape = cov.size() + self.y.size()[1:]
        cov = cov.view(cov_shape_pre).expand(cov_shape)

        return loc, cov
