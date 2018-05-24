from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel


def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


class Isotropy(Kernel):
    """
    Base class for a family of isotropic covariance kernels which are functions of the
    distance :math:`|x-z|/l`, where :math:`l` is the length-scale parameter.

    By default, the parameter ``lengthscale`` has size 1. To use the isotropic version
    (different lengthscale for each dimension), make sure that ``lengthscale`` has size
    equal to ``input_dim``.

    :param torch.Tensor lengthscale: Length-scale parameter of this kernel.
    """
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None,
                 name=None):
        super(Isotropy, self).__init__(input_dim, active_dims, name)

        if variance is None:
            variance = torch.tensor(1.)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

        if lengthscale is None:
            lengthscale = torch.tensor(1.)
        self.lengthscale = Parameter(lengthscale)
        self.set_constraint("lengthscale", constraints.positive)

    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.shape[1] != Z.shape[1]:
            raise ValueError("Inputs must have the same number of features.")

        lengthscale = self.get_param("lengthscale")
        scaled_X = X / lengthscale
        scaled_Z = Z / lengthscale
        X2 = (scaled_X ** 2).sum(1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0)

    def _scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))

    def _diag(self, X):
        """
        Calculates the diagonal part of covariance matrix on active features.
        """
        variance = self.get_param("variance")
        return variance.expand(X.shape[0])


class RBF(Isotropy):
    r"""
    Implementation of Radial Basis Function kernel:

        :math:`k(x,z) = \sigma^2\exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None,
                 name="RBF"):
        super(RBF, self).__init__(input_dim, variance, lengthscale, active_dims, name)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        r2 = self._square_scaled_dist(X, Z)
        return variance * torch.exp(-0.5 * r2)


class RationalQuadratic(Isotropy):
    r"""
    Implementation of RationalQuadratic kernel:

        :math:`k(x, z) = \sigma^2 \left(1 + 0.5 \times \frac{|x-z|^2}{\alpha l^2}
        \right)^{-\alpha}.`

    :param torch.Tensor scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """
    def __init__(self, input_dim, variance=None, lengthscale=None, scale_mixture=None,
                 active_dims=None, name="RationalQuadratic"):
        super(RationalQuadratic, self).__init__(input_dim, variance, lengthscale,
                                                active_dims, name)

        if scale_mixture is None:
            scale_mixture = torch.tensor(1.)
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
    r"""
    Implementation of Exponential kernel:

        :math:`k(x, z) = \sigma^2\exp\left(-\frac{|x-z|}{l}\right).`
    """
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None,
                 name="Exponential"):
        super(Exponential, self).__init__(input_dim, variance, lengthscale,
                                          active_dims, name)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        r = self._scaled_dist(X, Z)
        return variance * torch.exp(-r)


class Matern32(Isotropy):
    r"""
    Implementation of Matern32 kernel:

        :math:`k(x, z) = \sigma^2\left(1 + \sqrt{3} \times \frac{|x-z|}{l}\right)
        \exp\left(-\sqrt{3} \times \frac{|x-z|}{l}\right).`
    """
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None,
                 name="Matern32"):
        super(Matern32, self).__init__(input_dim, variance, lengthscale, active_dims,
                                       name)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        r = self._scaled_dist(X, Z)
        sqrt3_r = 3**0.5 * r
        return variance * (1 + sqrt3_r) * torch.exp(-sqrt3_r)


class Matern52(Isotropy):
    r"""
    Implementation of Matern52 kernel:

        :math:`k(x,z)=\sigma^2\left(1+\sqrt{5}\times\frac{|x-z|}{l}+\frac{5}{3}\times
        \frac{|x-z|^2}{l^2}\right)\exp\left(-\sqrt{5} \times \frac{|x-z|}{l}\right).`
    """
    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None,
                 name="Matern52"):
        super(Matern52, self).__init__(input_dim, variance, lengthscale, active_dims,
                                       name)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        r2 = self._square_scaled_dist(X, Z)
        r = _torch_sqrt(r2)
        sqrt5_r = 5**0.5 * r
        return variance * (1 + sqrt5_r + (5/3) * r2) * torch.exp(-sqrt5_r)
