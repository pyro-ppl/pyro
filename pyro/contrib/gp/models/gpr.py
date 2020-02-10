# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributions as torchdist
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.util import conditional
from pyro.nn.module import PyroParam, pyro_method
from pyro.util import warn_if_nan


class GPRegression(GPModel):
    r"""
    Gaussian Process Regression model.

    The core of a Gaussian Process is a covariance function :math:`k` which governs
    the similarity between input points. Given :math:`k`, we can establish a
    distribution over functions :math:`f` by a multivarite normal distribution

    .. math:: p(f(X)) = \mathcal{N}(0, k(X, X)),

    where :math:`X` is any set of input points and :math:`k(X, X)` is a covariance
    matrix whose entries are outputs :math:`k(x, z)` of :math:`k` over input pairs
    :math:`(x, z)`. This distribution is usually denoted by

    .. math:: f \sim \mathcal{GP}(0, k).

    .. note:: Generally, beside a covariance matrix :math:`k`, a Gaussian Process can
        also be specified by a mean function :math:`m` (which is a zero-value function
        by default). In that case, its distribution will be

        .. math:: p(f(X)) = \mathcal{N}(m(X), k(X, X)).

    Given inputs :math:`X` and their noisy observations :math:`y`, the Gaussian Process
    Regression model takes the form

    .. math::
        f &\sim \mathcal{GP}(0, k(X, X)),\\
        y & \sim f + \epsilon,

    where :math:`\epsilon` is Gaussian noise.

    .. note:: This model has :math:`\mathcal{O}(N^3)` complexity for training,
        :math:`\mathcal{O}(N^3)` complexity for testing. Here, :math:`N` is the number
        of train inputs.

    Reference:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor noise: Variance of Gaussian noise of this model.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, noise=None, mean_function=None, jitter=1e-6):
        super().__init__(X, y, kernel, mean_function, jitter)

        noise = self.X.new_tensor(1.) if noise is None else noise
        self.noise = PyroParam(noise, constraints.positive)

    @pyro_method
    def model(self):
        self.set_mode("model")

        N = self.X.size(0)
        Kff = self.kernel(self.X)
        Kff.view(-1)[::N + 1] += self.jitter + self.noise  # add noise to diagonal
        Lff = Kff.cholesky()

        zero_loc = self.X.new_zeros(self.X.size(0))
        f_loc = zero_loc + self.mean_function(self.X)
        if self.y is None:
            f_var = Lff.pow(2).sum(dim=-1)
            return f_loc, f_var
        else:
            return pyro.sample(self._pyro_get_fullname("y"),
                               dist.MultivariateNormal(f_loc, scale_tril=Lff)
                                   .expand_by(self.y.shape[:-1])
                                   .to_event(self.y.dim() - 1),
                               obs=self.y)

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

    def forward(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        N = self.X.size(0)
        Kff = self.kernel(self.X).contiguous()
        Kff.view(-1)[::N + 1] += self.jitter + self.noise  # add noise to the diagonal
        Lff = Kff.cholesky()

        y_residual = self.y - self.mean_function(self.X)
        loc, cov = conditional(Xnew, self.X, self.kernel, y_residual, None, Lff,
                               full_cov, jitter=self.jitter)

        if full_cov and not noiseless:
            M = Xnew.size(0)
            cov = cov.contiguous()
            cov.view(-1, M * M)[:, ::M + 1] += self.noise  # add noise to the diagonal
        if not full_cov and not noiseless:
            cov = cov + self.noise

        return loc + self.mean_function(Xnew), cov

    def iter_sample(self, noiseless=True):
        r"""
        Iteratively constructs a sample from the Gaussian Process posterior.

        Recall that at test input points :math:`X_{new}`, the posterior is
        multivariate Gaussian distributed with mean and covariance matrix
        given by :func:`forward`.

        This method samples lazily from this multivariate Gaussian. The advantage
        of this approach is that later query points can depend upon earlier ones.
        Particularly useful when the querying is to be done by an optimisation
        routine.

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

        :param bool noiseless: A flag to decide if we want to add sampling noise
            to the samples beyond the noise inherent in the GP posterior.
        :returns: sampler
        :rtype: function
        """
        noise = self.noise.detach()
        X = self.X.clone().detach()
        y = self.y.clone().detach()
        N = X.size(0)
        Kff = self.kernel(X).contiguous()
        Kff.view(-1)[::N + 1] += noise  # add noise to the diagonal

        outside_vars = {"X": X, "y": y, "N": N, "Kff": Kff}

        def sample_next(xnew, outside_vars):
            """Repeatedly samples from the Gaussian process posterior,
            conditioning on previously sampled values.
            """
            warn_if_nan(xnew)

            # Variables from outer scope
            X, y, Kff = outside_vars["X"], outside_vars["y"], outside_vars["Kff"]

            # Compute Cholesky decomposition of kernel matrix
            Lff = Kff.cholesky()
            y_residual = y - self.mean_function(X)

            # Compute conditional mean and variance
            loc, cov = conditional(xnew, X, self.kernel, y_residual, None, Lff,
                                   False, jitter=self.jitter)
            if not noiseless:
                cov = cov + noise

            ynew = torchdist.Normal(loc + self.mean_function(xnew), cov.sqrt()).rsample()

            # Update kernel matrix
            N = outside_vars["N"]
            Kffnew = Kff.new_empty(N+1, N+1)
            Kffnew[:N, :N] = Kff
            cross = self.kernel(X, xnew).squeeze()
            end = self.kernel(xnew, xnew).squeeze()
            Kffnew[N, :N] = cross
            Kffnew[:N, N] = cross
            # No noise, just jitter for numerical stability
            Kffnew[N, N] = end + self.jitter
            # Heuristic to avoid adding degenerate points
            if Kffnew.logdet() > -15.:
                outside_vars["Kff"] = Kffnew
                outside_vars["N"] += 1
                outside_vars["X"] = torch.cat((X, xnew))
                outside_vars["y"] = torch.cat((y, ynew))

            return ynew

        return lambda xnew: sample_next(xnew, outside_vars)
