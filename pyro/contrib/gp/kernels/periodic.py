from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import constraints
from torch.nn import Parameter

from .kernel import Kernel
from .isotropic import Isotropy


class Cosine(Isotropy):
    """
    Implementation of Cosine kernel :math:`\cos(r)`.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name="Cosine"):
        super(Cosine, self).__init__(input_dim, variance, lengthscale, active_dims, name)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        variance = self.get_param("variance")
        r = self._scaled_dist(X, Z)
        return variance * torch.cos(r)


class ExpSineSquared(Kernel):
    r"""
    Implementation of ExpSineSquared kernel (Periodic kernel)
    :math:`k(x, z) = \exp\left(-2 \times \frac{\sin^2(\pi (x-z) / p)}{l^2}\right)`,
    where :math:`p` is ``period`` parameter.

    References:

    [1] `Introduction to Gaussian processes`,
    David J.C. MacKay

    :param torch.Tensor variance: Variance parameter of this kernel.
    :param torch.Tensor lengthscale: Length scale parameter of this kernel.
    :param torch.Tensor period: Period parameter of this kernel.
    """
    def __init__(self, input_dim, variance=None, lengthscale=None, period=None, active_dims=None,
                 name="ExpSineSquared"):
        super(ExpSineSquared, self).__init__(input_dim, active_dims, name)

        if variance is None:
            variance = torch.tensor(1.)
        self.variance = Parameter(variance)
        self.set_constraint("variance", constraints.positive)

        if lengthscale is None:
            lengthscale = torch.tensor(1.)
        self.lengthscale = Parameter(lengthscale)
        self.set_constraint("lengthscale", constraints.positive)

        if period is None:
            period = torch.tensor(1.)
        self.period = Parameter(period)
        self.set_constraint("period", constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            variance = self.get_param("variance")
            return variance.expand(X.size(0))

        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        variance = self.get_param("variance")
        lengthscale = self.get_param("lengthscale")
        period = self.get_param("period")

        d = X.unsqueeze(1) - Z.unsqueeze(0)
        scaled_sin = torch.sin(math.pi * d / period) / lengthscale
        return variance * torch.exp(-2 * (scaled_sin ** 2).sum(-1))


class Periodic(ExpSineSquared):
    """
    Periodic is another name for :class:`ExpSineSquared` kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None,  period=None, active_dims=None,
                 name="Periodic"):
        super(Periodic, self).__init__(input_dim, variance, lengthscale, period, active_dims, name)
