from __future__ import absolute_import, division, print_function

from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.util import conditional

from .model import GPModel


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

    Given inputs :math:`X` and their noisy observations :math:`y`, the Gaussian Process
    Regression model takes the form

    .. math::
        f &\sim \mathcal{GP}(0, k(X, X)),\\
        y & \sim f + \epsilon,

    where :math:`\epsilon` is noise.

    .. note:: This model has :math:`\mathcal{O}(N^3)` complexity for training,
        :math:`\mathcal{O}(N^3)` complexity for testing. Here, :math:`N` is the number
        of train inputs.

    Reference:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor X: A 1D or 2D input data for training. Its first dimension is
        the number of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor noise: Noise parameter of this model.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :param str name: Name of this model.
    """
    def __init__(self, X, y, kernel, noise=None, jitter=1e-6, name="GPR"):
        super(GPRegression, self).__init__(X, y, kernel, jitter, name)

        noise = self.X.new_ones(()) if noise is None else noise
        self.noise = Parameter(noise)
        self.set_constraint("noise", constraints.greater_than(self.jitter))

    def model(self):
        self.set_mode("model")

        noise = self.get_param("noise")

        Kff = self.kernel(self.X) + noise.expand(self.X.shape[0]).diag()
        Lff = Kff.potrf(upper=False)

        zero_loc = self.X.new_zeros(self.X.shape[0])
        if self.y is None:
            f_var = Lff.pow(2).sum(dim=-1)
            return zero_loc, f_var
        else:
            y_name = pyro.param_with_module_name(self.name, "y")
            return pyro.sample(y_name,
                               dist.MultivariateNormal(zero_loc, scale_tril=Lff)
                                   .reshape(sample_shape=self.y.shape[:-1],
                                            extra_event_dims=self.y.dim()-1),
                               obs=self.y)

    def guide(self):
        self.set_mode("guide")

        kernel = self.kernel
        noise = self.get_param("noise")

        return kernel, noise

    def forward(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

        :param torch.Tensor Xnew: A 1D or 2D input data for testing. In 2D case, its
            second dimension should have the same size as of train input data.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        kernel, noise = self.guide()

        Kff = kernel(self.X) + noise.expand(self.X.shape[0]).diag()
        Lff = Kff.potrf(upper=False)

        loc, cov = conditional(Xnew, self.X, kernel, self.y, None, Lff, full_cov,
                               jitter=self.jitter)

        if full_cov and not noiseless:
            cov = cov + noise.expand(Xnew.shape[0]).diag()
        if not full_cov and not noiseless:
            cov = cov + noise.expand(Xnew.shape[0])

        return loc, cov
