# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.util import conditional
from pyro.distributions.util import eye_like
from pyro.nn.module import PyroParam, pyro_method


class VariationalSparseGP(GPModel):
    r"""
    Variational Sparse Gaussian Process model.

    In :class:`.VariationalGP` model, when the number of input data :math:`X` is large,
    the covariance matrix :math:`k(X, X)` will require a lot of computational steps to
    compute its inverse (for log likelihood and for prediction). This model introduces
    an additional inducing-input parameter :math:`X_u` to solve that problem. Given
    inputs :math:`X`, their noisy observations :math:`y`, and the inducing-input
    parameters :math:`X_u`, the model takes the form:

    .. math::
        [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
        y & \sim p(y) = p(y \mid f) p(f),

    where :math:`p(y \mid f)` is the likelihood.

    We will use a variational approach in this model by approximating :math:`q(f,u)`
    to the posterior :math:`p(f,u \mid y)`. Precisely, :math:`q(f) = p(f\mid u)q(u)`,
    where :math:`q(u)` is a multivariate normal distribution with two parameters
    ``u_loc`` and ``u_scale_tril``, which will be learned during a variational
    inference process.

    .. note:: This model can be learned using MCMC method as in reference [2]. See also
        :class:`.GPModel`.

    .. note:: This model has :math:`\mathcal{O}(NM^2)` complexity for training,
        :math:`\mathcal{O}(M^3)` complexity for testing. Here, :math:`N` is the number
        of train inputs, :math:`M` is the number of inducing inputs. Size of
        variational parameters is :math:`\mathcal{O}(M^2)`.

    References:

    [1] `Scalable variational Gaussian process classification`,
    James Hensman, Alexander G. de G. Matthews, Zoubin Ghahramani

    [2] `MCMC for Variationally Sparse Gaussian Processes`,
    James Hensman, Alexander G. de G. Matthews, Maurizio Filippone, Zoubin Ghahramani

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor Xu: Initial values for inducing points, which are parameters
        of our model.
    :param ~pyro.contrib.gp.likelihoods.likelihood Likelihood likelihood: A likelihood
        object.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param torch.Size latent_shape: Shape for latent processes (`batch_shape` of
        :math:`q(u)`). By default, it equals to output batch shape ``y.shape[:-1]``.
        For the multi-class classification problems, ``latent_shape[-1]`` should
        corresponse to the number of classes.
    :param int num_data: The size of full training dataset. It is useful for training
        this model with mini-batch.
    :param bool whiten: A flag to tell if variational parameters ``u_loc`` and
        ``u_scale_tril`` are transformed by the inverse of ``Luu``, where ``Luu`` is
        the lower triangular decomposition of :math:`kernel(X_u, X_u)`. Enable this
        flag will help optimization.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, Xu, likelihood, mean_function=None,
                 latent_shape=None, num_data=None, whiten=False, jitter=1e-6):
        super().__init__(X, y, kernel, mean_function, jitter)

        self.likelihood = likelihood
        self.Xu = Parameter(Xu)

        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        M = self.Xu.size(0)
        u_loc = self.Xu.new_zeros(self.latent_shape + (M,))
        self.u_loc = Parameter(u_loc)

        identity = eye_like(self.Xu, M)
        u_scale_tril = identity.repeat(self.latent_shape + (1, 1))
        self.u_scale_tril = PyroParam(u_scale_tril, constraints.lower_cholesky)

        self.num_data = num_data if num_data is not None else self.X.size(0)
        self.whiten = whiten
        self._sample_latent = True

    @pyro_method
    def model(self):
        self.set_mode("model")

        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
        Luu = Kuu.cholesky()

        zero_loc = self.Xu.new_zeros(self.u_loc.shape)
        if self.whiten:
            identity = eye_like(self.Xu, M)
            pyro.sample(self._pyro_get_fullname("u"),
                        dist.MultivariateNormal(zero_loc, scale_tril=identity)
                            .to_event(zero_loc.dim() - 1))
        else:
            pyro.sample(self._pyro_get_fullname("u"),
                        dist.MultivariateNormal(zero_loc, scale_tril=Luu)
                            .to_event(zero_loc.dim() - 1))

        f_loc, f_var = conditional(self.X, self.Xu, self.kernel, self.u_loc, self.u_scale_tril,
                                   Luu, full_cov=False, whiten=self.whiten, jitter=self.jitter)

        f_loc = f_loc + self.mean_function(self.X)
        if self.y is None:
            return f_loc, f_var
        else:
            # we would like to load likelihood's parameters outside poutine.scale context
            self.likelihood._load_pyro_samples()
            with poutine.scale(scale=self.num_data / self.X.size(0)):
                return self.likelihood(f_loc, f_var, self.y)

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

        pyro.sample(self._pyro_get_fullname("u"),
                    dist.MultivariateNormal(self.u_loc, scale_tril=self.u_scale_tril)
                        .to_event(self.u_loc.dim()-1))

    def forward(self, Xnew, full_cov=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).

        .. note:: Variational parameters ``u_loc``, ``u_scale_tril``, the
            inducing-point parameter ``Xu``, together with kernel's parameters have
            been learned from a training procedure (MCMC or SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        loc, cov = conditional(Xnew, self.Xu, self.kernel, self.u_loc, self.u_scale_tril,
                               full_cov=full_cov, whiten=self.whiten, jitter=self.jitter)
        return loc + self.mean_function(Xnew), cov
