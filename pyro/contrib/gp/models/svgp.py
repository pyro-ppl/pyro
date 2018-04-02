from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.util import conditional

from .model import GPModel


class SparseVariationalGP(GPModel):
    """
    Sparse Variational Gaussian Process model.

    This model introduces an additional inducing-input parameter :math:`X_u` as in
    :class:`~.SparseGPRegression` model and deals with both Gaussian and non-Gaussian
    likelihoods. Given inputs :math:`X`, their noisy observations :math:`y`, and the
    inducing-input parameters :math:`X_u`, the model takes the form:

    .. math::
        [f, u] \sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
        y & \sim p(y) = p(y \mid f) p(f),

    where :math:`p(y \mid f)` is the likelihood.

    We will use a variational approach in this model by approximating `q(f,u)` to the
    posterior :math:`p(f,u \mid y)`. Precisely, :math:`q(f) = p(f\mid u)q(u)`,
    where :math:`q(u)` is a multivariate normal distribution with two parameters
    ``f_loc`` and ``f_scale_tril``, which will be learned during a variational
    inference process.

    .. note:: This model can be learned using MCMC method as in reference [2].

    References

    [1] `Scalable variational Gaussian process classification`,
    James Hensman, Alexander G. de G. Matthews, Zoubin Ghahramani

    [2] `MCMC for Variationally Sparse Gaussian Processes`,
    James Hensman, Alexander G. de G. Matthews, Maurizio Filippone, Zoubin Ghahramani

    :param torch.Tensor X: A 1D or 2D input data for training. Its first dimension is
        the number of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor Xu: Initial values for inducing points, which are parameters
        of our model.
    :param ~pyro.contrib.gp.likelihoods.likelihood Likelihood likelihood: A likelihood
        object.
    :param torch.Size latent_shape: Shape for latent processes (`batch_shape` of
        :math:`q(u)`). By default, it equals to output batch shape ``y.shape[:-1]``.
        For the multi-class classification problems, ``latent_shape[-1]`` should
        corresponse to the number of classes.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :param str name: Name of this model.
    """
    def __init__(self, X, y, kernel, Xu, likelihood, latent_shape=None,
                 jitter=1e-6, name="SVGP"):
        super(SparseVariationalGP, self).__init__(X, y, kernel, jitter, name)
        self.likelihood = likelihood

        self.Xu = Parameter(Xu)

        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        M = self.Xu.shape[0]
        u_loc_shape = self.latent_shape + (M,)
        u_loc = self.Xu.new_zeros(u_loc_shape)
        self.u_loc = Parameter(u_loc)

        u_scale_tril_shape = self.latent_shape + (M, M)
        u_scale_tril = torch.eye(M, out=self.Xu.new(M, M))
        u_scale_tril = u_scale_tril.expand(u_scale_tril_shape)
        self.u_scale_tril = Parameter(u_scale_tril)
        self.set_constraint("u_scale_tril", constraints.lower_cholesky)

        self._sample_latent = True

    def model(self):
        self.set_mode("model")

        Xu = self.get_param("Xu")
        u_loc = self.get_param("u_loc")
        u_scale_tril = self.get_param("u_scale_tril")

        M = Xu.shape[0]
        Kuu = self.kernel(Xu) + torch.eye(M, out=Xu.new(M, M)) * self.jitter
        Luu = Kuu.potrf(upper=False)

        zero_loc = Xu.new_zeros(u_loc.shape)
        u_name = pyro.param_with_module_name(self.name, "u")
        pyro.sample(u_name,
                    dist.MultivariateNormal(zero_loc, scale_tril=Luu)
                        .reshape(extra_event_dims=zero_loc.dim()-1))

        f_loc, f_var = conditional(self.X, Xu, self.kernel, u_loc, u_scale_tril,
                                   Luu, full_cov=False, jitter=self.jitter)

        if self.y is None:
            return f_loc, f_var
        else:
            return self.likelihood(f_loc, f_var, self.y)

    def guide(self):
        self.set_mode("guide")

        Xu = self.get_param("Xu")
        u_loc = self.get_param("u_loc")
        u_scale_tril = self.get_param("u_scale_tril")

        if self._sample_latent:
            u_name = pyro.param_with_module_name(self.name, "u")
            pyro.sample(u_name,
                        dist.MultivariateNormal(u_loc, scale_tril=u_scale_tril)
                            .reshape(extra_event_dims=u_loc.dim()-1))
        return Xu, self.kernel, u_loc, u_scale_tril

    def forward(self, Xnew, full_cov=False):
        self._check_Xnew_shape(Xnew)
        tmp_sample_latent = self._sample_latent
        self._sample_latent = False
        Xu, kernel, u_loc, u_scale_tril = self.guide()
        self._sample_latent = tmp_sample_latent

        loc, cov = conditional(Xnew, Xu, kernel, u_loc, u_scale_tril,
                               full_cov=full_cov, jitter=self.jitter)
        return loc, cov
