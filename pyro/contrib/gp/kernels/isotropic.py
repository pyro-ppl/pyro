from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel


def _torch_sqrt(x, eps=1e-18):
    """
    A convenient function to avoid the NaN gradient issue of ``torch.sqrt`` at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


class Isotropy(Kernel):
    """
    Base kernel for a family of isotropic covariance functions which is a
    function of the distance ``r=|x-z|``.

    By default, the parameter ``lengthscale`` has size 1. To use the
    anisotropic version (different lengthscale for each dimension),
    make sure that lengthscale has size equal to ``input_dim``.

    :param torch.Tensor variance: Variance parameter of this kernel.
    :param torch.Tensor lengthscale: Length scale parameter of this kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name=None):
        super(Isotropy, self).__init__(input_dim, active_dims, name)

        if variance is None:
            variance = torch.ones(1)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

        if lengthscale is None:
            lengthscale = torch.ones(1)
        self.lengthscale = Parameter(lengthscale)
        self.set_constraint("lengthscale", constraints.positive)

    def _square_scaled_dist(self, X, Z=None):
        """
        Returns ``||(X-Z)/lengthscale||^2``.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        lengthscale = self.get_param("lengthscale")
        scaled_X = X / lengthscale
        scaled_Z = Z / lengthscale
        X2 = (scaled_X ** 2).sum(1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        return r2

    def _scaled_dist(self, X, Z=None):
        """
        Returns ``||(X-Z)/lengthscale||``.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))

    def _diag(self, X):
        """
        Calculates the diagonal part of covariance matrix on active dimensionals.
        """
        variance = self.get_param("variance")
        return variance.expand(X.size(0))


class RBF(Isotropy):
    """
    Implementation of Radial Basis Function kernel: ``exp(-0.5 * r^2 / l^2)``.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name="RBF"):
        super(RBF, self).__init__(input_dim, variance, lengthscale, active_dims, name)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        r2 = self._square_scaled_dist(X, Z)
        return variance * torch.exp(-0.5 * r2)


class SquaredExponential(RBF):
    """
    Squared Exponential is another name for RBF kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None,
                 name="SquaredExponential"):
        super(SquaredExponential, self).__init__(input_dim, variance, lengthscale, active_dims, name)


class RationalQuadratic(Isotropy):
    """
    Implementation of Rational Quadratic kernel: ``(1 + 0.5 * r^2 / alpha l^2)^(-alpha)``.

    :param torch.Tensor scale_mixture: Scale mixture (alpha) parameter of this kernel.
        Should have size 1.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, scale_mixture=None, active_dims=None,
                 name="RationalQuadratic"):
        super(RationalQuadratic, self).__init__(input_dim, variance, lengthscale, active_dims, name)

        if scale_mixture is None:
            scale_mixture = torch.ones(1)
        self.scale_mixture = Parameter(scale_mixture)
        self.set_constraint("scale_mixture", constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        scale_mixture = self.get_param("scale_mixture")
        r2 = self._square_scaled_dist(X, Z)
        return variance * (1 + (0.5 / scale_mixture) * r2).pow(-scale_mixture)


class Exponential(Isotropy):
    """
    Implementation of Exponential kernel: `exp(-r/l)`.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name="Exponential"):
        super(Exponential, self).__init__(input_dim, variance, lengthscale, active_dims, name)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        r = self._scaled_dist(X, Z)
        return variance * torch.exp(-r)


class Matern12(Exponential):
    """
    Another name of Exponential kernel.
    """
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name="Matern12"):
        super(Matern12, self).__init__(input_dim, variance, lengthscale, active_dims, name)


class Matern32(Isotropy):
    """
    Implementation of Matern32 kernel: ``(1 + sqrt(3) * r/l) * exp(-sqrt(3) * r/l)``.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name="Matern32"):
        super(Matern32, self).__init__(input_dim, variance, lengthscale, active_dims, name)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        r = self._scaled_dist(X, Z)
        sqrt3_r = 3**0.5 * r
        return variance * (1 + sqrt3_r) * torch.exp(-sqrt3_r)


class Matern52(Isotropy):
    """
    Implementation of Matern52 kernel: ``(1 + sqrt(5) * r/l + 5/3 * r^2 / l^2) * exp(-sqrt(5) * r/l)``.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name="Matern52"):
        super(Matern52, self).__init__(input_dim, variance, lengthscale, active_dims, name)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        r2 = self._square_scaled_dist(X, Z)
        r = _torch_sqrt(r2)
        sqrt5_r = 5**0.5 * r
        return variance * (1 + sqrt5_r + (5/3) * r2) * torch.exp(-sqrt5_r)
