from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


def _matrix_inverse_compat(matrix, matrix_chol):
    """Computes the inverse of a positive semidefinite square matrix."""
    if matrix.requires_grad:
        # If derivatives are required, use the more expensive inverse.
        return torch.inverse(matrix)
    else:
        # Use the cheaper Cholesky based potri.
        return torch.potri(matrix_chol)


# TODO Move this upstream to PyTorch.
class TorchMultivariateNormal(torch.distributions.Distribution):
    params = {"loc": constraints.real, "scale_tril": constraints.lower_triangular}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, covariance_matrix, normalized=True):
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        batch_shape, event_shape = loc.shape[:-1], loc.shape[-1:]
        super(TorchMultivariateNormal, self).__init__(batch_shape, event_shape)

    @lazy_property
    def scale_triu(self):
        return torch.potrf(self.covariance_matrix)

    def rsample(self, sample_shape=torch.Size()):
        white = self.loc.new(sample_shape + self.loc.shape).normal_()
        return self.loc + torch.matmul(white, self.scale_triu)

    def log_prob(self, value):
        delta = value - self.loc
        sigma_inverse = _matrix_inverse_compat(self.covariance_matrix, self.scale_triu)
        normalization_const = ((0.5 * self.event_shape[-1]) * math.log(2 * math.pi) +
                               self.scale_triu.diag().log().sum(-1))
        mahalanobis_squared = (delta * torch.matmul(delta, sigma_inverse)).sum(-1)
        return -(normalization_const + 0.5 * mahalanobis_squared)


@copy_docs_from(TorchDistribution)
class MultivariateNormal(TorchDistribution):
    """Multivariate normal (Gaussian) distribution.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.autograd.Variable loc: Mean.
    :param torch.autograd.Variable covariance_matrix: Covariance matrix.
        Must be symmetric and positive semidefinite.
    """
    reparameterized = True

    def __init__(self, loc, covariance_matrix, *args, **kwargs):
        torch_dist = TorchMultivariateNormal(loc, covariance_matrix)
        super(MultivariateNormal, self).__init__(torch_dist, *args, **kwargs)
