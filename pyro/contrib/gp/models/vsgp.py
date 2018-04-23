from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.util import conditional
from pyro.params import param_with_module_name


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
    :param str name: Name of this model.
    """
    def __init__(self, X, y, kernel, Xu, likelihood, mean_function=None,
                 latent_shape=None, num_data=None, whiten=False, jitter=1e-6,
                 name="SVGP"):
        super(VariationalSparseGP, self).__init__(X, y, kernel, mean_function, jitter,
                                                  name)
        self.likelihood = likelihood

        self.num_data = num_data if num_data is not None else self.X.shape[0]
        self.whiten = whiten

        self.Xu = Parameter(Xu)

        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        M = self.Xu.shape[0]
        u_loc_shape = self.latent_shape + (M,)
        u_loc = self.Xu.new_zeros(u_loc_shape)
        self.u_loc = Parameter(u_loc)

        u_scale_tril_shape = self.latent_shape + (M, M)
        Id = torch.eye(M, out=self.Xu.new_empty(M, M))
        u_scale_tril = Id.expand(u_scale_tril_shape)
        self.u_scale_tril = Parameter(u_scale_tril)
        self.set_constraint("u_scale_tril", constraints.lower_cholesky)

        self._sample_latent = True

    def model(self):
        self.set_mode("model")

        Xu = self.get_param("Xu")
        u_loc = self.get_param("u_loc")
        u_scale_tril = self.get_param("u_scale_tril")

        M = Xu.shape[0]
        Kuu = self.kernel(Xu) + torch.eye(M, out=Xu.new_empty(M, M)) * self.jitter
        Luu = Kuu.potrf(upper=False)

        zero_loc = Xu.new_zeros(u_loc.shape)
        u_name = param_with_module_name(self.name, "u")
        if self.whiten:
            Id = torch.eye(M, out=Xu.new_empty(M, M))
            pyro.sample(u_name,
                        dist.MultivariateNormal(zero_loc, scale_tril=Id)
                            .independent(zero_loc.dim() - 1))
        else:
            pyro.sample(u_name,
                        dist.MultivariateNormal(zero_loc, scale_tril=Luu)
                            .independent(zero_loc.dim() - 1))

        f_loc, f_var = conditional(self.X, Xu, self.kernel, u_loc, u_scale_tril,
                                   Luu, full_cov=False, whiten=self.whiten,
                                   jitter=self.jitter)

        f_loc = f_loc + self.mean_function(self.X)
        if self.y is None:
            return f_loc, f_var
        else:
            with poutine.scale(None, self.num_data / self.X.shape[0]):
                return self.likelihood(f_loc, f_var, self.y)

    def guide(self):
        self.set_mode("guide")

        Xu = self.get_param("Xu")
        u_loc = self.get_param("u_loc")
        u_scale_tril = self.get_param("u_scale_tril")

        if self._sample_latent:
            u_name = param_with_module_name(self.name, "u")
            pyro.sample(u_name,
                        dist.MultivariateNormal(u_loc, scale_tril=u_scale_tril)
                            .independent(u_loc.dim()-1))
        return Xu, u_loc, u_scale_tril

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
        # avoid sampling the unnecessary latent u
        self._sample_latent = False
        Xu, u_loc, u_scale_tril = self.guide()
        self._sample_latent = True

        loc, cov = conditional(Xnew, Xu, self.kernel, u_loc, u_scale_tril,
                               full_cov=full_cov, whiten=self.whiten,
                               jitter=self.jitter)
        return loc + self.mean_function(Xnew), cov
