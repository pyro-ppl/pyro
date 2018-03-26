from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist

from .model import GPModel


class GPRegression(GPModel):
    """
    Gaussian Process Regression module.

    References

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor X: A 1D or 2D tensor of input data for training.
    :param torch.Tensor y: A tensor of output data for training with
        ``y.shape[-1]`` equals to number of data points.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor noise: An optional noise parameter.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, noise=None, jitter=1e-6):
        latent_shape = torch.Size([])
        super(GPRegression, self).__init__(X, y, kernel, latent_shape, jitter)

        noise = self.X.new_ones(1) if noise is None else noise
        self.noise = Parameter(noise)
        self.set_constraint("noise", constraints.greater_than(self.jitter))

    def model(self):
        self.set_mode("model")

        noise = self.get_param("noise")

        Kff = self.kernel(self.X) + noise.expand(self.X.shape[0]).diag()
        Lff = Kff.potrf(upper=False)

        if self.y is None:
            zero_loc = self.X.new_zeros(self.X.shape[0])
            return pyro.sample("y", dist.MultivariateNormal(zero_loc, scale_tril=Lff))
        else:
            zero_loc = self.X.new_zeros(self.y.shape)
            return pyro.sample("y", dist.MultivariateNormal(zero_loc, scale_tril=Lff)
                               .reshape(extra_event_dims=self.y.dim()-1), obs=self.y)

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
        self._check_Xnew_shape(Xnew)
        kernel, noise = self.guide()

        Kff = kernel(self.X) + noise.expand(self.X.shape[0]).diag()
        Lff = Kff.potrf(upper=False)

        loc, cov = self._conditional(Xnew, self.X, kernel, self.y, f_scale_tril=None,
                                     Lff=Lff, full_cov=full_cov)

        if full_cov and not noiseless:
            cov = cov + noise.expand(Xnew.shape[0]).diag()
        if not full_cov and not noiseless:
            cov = cov + noise.expand(Xnew.shape[0])

        return loc, cov
