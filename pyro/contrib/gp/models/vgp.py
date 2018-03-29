from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.util import conditional

from .model import GPModel


class VariationalGP(GPModel):
    """
    Variational Gaussian Process module.

    This model can be seen as a special version of SparseVariationalGP model
    with :math:`Xu = X`.

    :param torch.Tensor X: A 1D or 2D tensor of inputs.
    :param torch.Tensor y: A tensor of output data for training with
        ``y.shape[-1]`` equals to number of data points.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param torch.Size latent_shape: Shape for latent processes. By default, it equals
        to output batch shape ``y.shape[:-1]``. For the multi-class classification
        problems, ``latent_shape[-1]`` should corresponse to the number of classes.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, likelihood, latent_shape=None,
                 jitter=1e-6, name="VGP"):
        super(VariationalGP, self).__init__(X, y, kernel, jitter, name)
        self.likelihood = likelihood

        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        N = self.X.shape[0]
        f_loc_shape = self.latent_shape + (N,)
        f_loc = self.X.new_zeros(f_loc_shape)
        self.f_loc = Parameter(f_loc)

        f_scale_tril_shape = self.latent_shape + (N, N)
        f_scale_tril = torch.eye(N, out=self.X.new(N, N))
        f_scale_tril = f_scale_tril.expand(f_scale_tril_shape)
        self.f_scale_tril = Parameter(f_scale_tril)
        self.set_constraint("f_scale_tril", constraints.lower_cholesky)

        self._sample_latent = True

    def model(self):
        self.set_mode("model")

        f_loc = self.get_param("f_loc")
        f_scale_tril = self.get_param("f_scale_tril")

        N = self.X.shape[0]
        Kff = self.kernel(self.X) + torch.eye(N, out=self.X.new(N, N)) * self.jitter
        Lff = Kff.potrf(upper=False)

        zero_loc = self.X.new_zeros(f_loc.shape)
        f_name = pyro.param_with_module_name(self.name, "f")
        pyro.sample(f_name,
                    dist.MultivariateNormal(zero_loc, scale_tril=Lff)
                        .reshape(extra_event_dims=zero_loc.dim()-1))

        f_var = f_scale_tril.pow(2).sum(dim=-1)

        if self.y is None:
            return f_loc, f_var
        else:
            return self.likelihood(f_loc, f_var, self.y)

    def guide(self):
        self.set_mode("guide")

        f_loc = self.get_param("f_loc")
        f_scale_tril = self.get_param("f_scale_tril")

        if self._sample_latent:
            f_name = pyro.param_with_module_name(self.name, "f")
            pyro.sample(f_name,
                        dist.MultivariateNormal(f_loc, scale_tril=f_scale_tril)
                            .reshape(extra_event_dims=f_loc.dim()-1))
        return self.kernel, f_loc, f_scale_tril

    def forward(self, Xnew, full_cov=False):
        """
        Computes the parameters of :math:`p(f^*|Xnew) \sim N(\\text{loc}, \\text{cov})`
        according to :math:`p(f^*,f|y) = p(f^*|f)p(f|y) \sim p(f^*|f)q(f)`, then
        marginalize out variable :math:`f`. In case output data is a 2D tensor of shape
        :math:`N \times D`, :math:`loc` is also a 2D tensor of shape :math:`N \times D`.
        Covariance matrix :math:`cov` is always a 2D tensor of shape :math:`N \times N`.

        :param torch.Tensor Xnew: A 1D or 2D tensor.
        :param bool full_cov: Predict full covariance matrix or just its diagonal.
        :returns: loc and covariance matrix of :math:`p(f^*|Xnew)`
        :rtype: torch.Tensor and torch.Tensor
        """
        self._check_Xnew_shape(Xnew)
        tmp_sample_latent = self._sample_latent
        self._sample_latent = False
        kernel, f_loc, f_scale_tril = self.guide()
        self._sample_latent = tmp_sample_latent

        loc, cov = conditional(Xnew, self.X, kernel, f_loc, f_scale_tril,
                               full_cov=full_cov, jitter=self.jitter)
        return loc, cov
