from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch_distribution import TorchDistribution


def _matrix_inverse_compat(matrix, matrix_chol):
    """Computes the inverse of a positive semidefinite square matrix."""
    if matrix.requires_grad:
        # If derivatives are required, use the more expensive inverse.
        return torch.inverse(matrix)
    else:
        # Use the cheaper Cholesky based potri.
        return torch.potri(matrix_chol, upper=False)


# TODO Move this upstream to PyTorch.
class MultivariateNormal(TorchDistribution):
    """Multivariate normal (Gaussian) distribution.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.Tensor loc: Mean.
    :param torch.Tensor covariance_matrix: Covariance matrix.
        Must be symmetric and positive semidefinite.
    """
    params = {"loc": constraints.real, "scale_tril": constraints.lower_triangular}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, covariance_matrix, normalized=True):
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        batch_shape, event_shape = loc.shape[:-1], loc.shape[-1:]
        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.covariance_matrix.diag()

    @lazy_property
    def scale_tril(self):
        return torch.potrf(self.covariance_matrix, upper=False)

    def rsample(self, sample_shape=torch.Size()):
        white = self.loc.new(sample_shape + self.loc.shape).normal_()
        return self.loc + torch.matmul(white, self.scale_tril.t())

    def log_prob(self, value):
        delta = value - self.loc
        sigma_inverse = _matrix_inverse_compat(self.covariance_matrix, self.scale_tril)
        normalization_const = ((0.5 * self.event_shape[-1]) * math.log(2 * math.pi) +
                               self.scale_tril.diag().log().sum(-1))
        mahalanobis_squared = (delta * torch.matmul(delta, sigma_inverse)).sum(-1)
        return -(normalization_const + 0.5 * mahalanobis_squared)
