from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.util import conditional
from pyro.params import param_with_module_name


class VariationalGP(GPModel):
    r"""
    Variational Gaussian Process model.

    This model deals with both Gaussian and non-Gaussian likelihoods. Given inputs\
    :math:`X` and their noisy observations :math:`y`, the model takes the form

    .. math::
        f &\sim \mathcal{GP}(0, k(X, X)),\\
        y & \sim p(y) = p(y \mid f) p(f),

    where :math:`p(y \mid f)` is the likelihood.

    We will use a variational approach in this model by approximating :math:`q(f)` to
    the posterior :math:`p(f\mid y)`. Precisely, :math:`q(f)` will be a multivariate
    normal distribution with two parameters ``f_loc`` and ``f_scale_tril``, which will
    be learned during a variational inference process.

    .. note:: This model can be seen as a special version of
        :class:`.SparseVariationalGP` model with :math:`X_u = X`.

    .. note:: This model has :math:`\mathcal{O}(N^3)` complexity for training,
        :math:`\mathcal{O}(N^3)` complexity for testing. Here, :math:`N` is the number
        of train inputs. Size of variational parameters is :math:`\mathcal{O}(N^2)`.

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param ~pyro.contrib.gp.likelihoods.likelihood Likelihood likelihood: A likelihood
        object.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param torch.Size latent_shape: Shape for latent processes (`batch_shape` of
        :math:`q(f)`). By default, it equals to output batch shape ``y.shape[:-1]``.
        For the multi-class classification problems, ``latent_shape[-1]`` should
        corresponse to the number of classes.
    :param bool whiten: A flag to tell if variational parameters ``f_loc`` and
        ``f_scale_tril`` are transformed by the inverse of ``Lff``, where ``Lff`` is
        the lower triangular decomposition of :math:`kernel(X, X)`. Enable this flag
        will help optimization.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :param str name: Name of this model.
    """
    def __init__(self, X, y, kernel, likelihood, mean_function=None,
                 latent_shape=None, whiten=False, jitter=1e-6, name="VGP"):
        super(VariationalGP, self).__init__(X, y, kernel, mean_function, jitter, name)
        self.likelihood = likelihood

        self.whiten = whiten

        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        N = self.X.shape[0]
        f_loc_shape = self.latent_shape + (N,)
        f_loc = self.X.new_zeros(f_loc_shape)
        self.f_loc = Parameter(f_loc)

        f_scale_tril_shape = self.latent_shape + (N, N)
        Id = torch.eye(N, out=self.X.new_empty(N, N))
        f_scale_tril = Id.expand(f_scale_tril_shape)
        self.f_scale_tril = Parameter(f_scale_tril)
        self.set_constraint("f_scale_tril", constraints.lower_cholesky)

        self._sample_latent = True

    def model(self):
        self.set_mode("model")

        f_loc = self.get_param("f_loc")
        f_scale_tril = self.get_param("f_scale_tril")

        N = self.X.shape[0]
        Kff = self.kernel(self.X) + (torch.eye(N, out=self.X.new_empty(N, N)) *
                                     self.jitter)
        Lff = Kff.potrf(upper=False)

        zero_loc = self.X.new_zeros(f_loc.shape)
        f_name = param_with_module_name(self.name, "f")

        if self.whiten:
            Id = torch.eye(N, out=self.X.new_empty(N, N))
            pyro.sample(f_name,
                        dist.MultivariateNormal(zero_loc, scale_tril=Id)
                            .independent(zero_loc.dim() - 1))
            f_scale_tril = Lff.matmul(f_scale_tril)
        else:
            pyro.sample(f_name,
                        dist.MultivariateNormal(zero_loc, scale_tril=Lff)
                            .independent(zero_loc.dim() - 1))

        f_var = f_scale_tril.pow(2).sum(dim=-1)

        if self.whiten:
            f_loc = Lff.matmul(f_loc.unsqueeze(-1)).squeeze(-1)
        f_loc = f_loc + self.mean_function(self.X)
        if self.y is None:
            return f_loc, f_var
        else:
            return self.likelihood(f_loc, f_var, self.y)

    def guide(self):
        self.set_mode("guide")

        f_loc = self.get_param("f_loc")
        f_scale_tril = self.get_param("f_scale_tril")

        if self._sample_latent:
            f_name = param_with_module_name(self.name, "f")
            pyro.sample(f_name,
                        dist.MultivariateNormal(f_loc, scale_tril=f_scale_tril)
                            .independent(f_loc.dim()-1))
        return f_loc, f_scale_tril

    def forward(self, Xnew, full_cov=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, f_{loc}, f_{scale\_tril})
            = \mathcal{N}(loc, cov).

        .. note:: Variational parameters ``f_loc``, ``f_scale_tril``, together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        # avoid sampling the unnecessary latent f
        self._sample_latent = False
        f_loc, f_scale_tril = self.guide()
        self._sample_latent = True

        loc, cov = conditional(Xnew, self.X, self.kernel, f_loc, f_scale_tril,
                               full_cov=full_cov, whiten=self.whiten,
                               jitter=self.jitter)
        return loc + self.mean_function(Xnew), cov
