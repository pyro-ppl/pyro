from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution


class MultivariateNormal(Distribution):
    """
    Multivariate normal (Gaussian) distribution.

    A distribution over vectors in which all the elements have a joint
    Gaussian distribution.

    :param torch.autograd.Variable mu: Mean.
    :param torch.autograd.Variable sigma: Covariance matrix.
        Must be symmetric and positive semidefinite.
    :param is_cholesky: Should be set to True if you want to directly pass a cholesky decomposition as sigma.
    """

    def __init__(self, mu, sigma, batch_size=None, is_cholesky = False, use_inverse_for_batch_log=False, *args, **kwargs):
        self.mu = mu
        self.output_shape = mu.shape
        self.use_inverse_for_batch_log = use_inverse_for_batch_log
        if not is_cholesky:
            self.sigma = sigma
            # potrf is the very sensible name for the Cholesky decomposition in PyTorch
            self.sigma_cholesky = torch.potrf(sigma)
        else:
            self.sigma = sigma.transpose(0,1) @ sigma
            self.sigma_cholesky = sigma
        if mu.dim() > 1:
            raise ValueError("The mean must be a vector, but got mu.size() = {}".format(mu.size()))

        super(MultivariateNormal, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        raise NotImplementedError()

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        raise NotImplementedError()

    def sample(self):
        """
        A classic multivariate normal sampler.

        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        uncorrelated_standard_sample = Variable(torch.randn(self.mu.size()).type_as(self.mu.data))
        transformed_sample = self.mu + uncorrelated_standard_sample @ self.sigma_cholesky
        return transformed_sample

    def batch_log_pdf(self, x):
        normalization_factor = 0.5 * torch.log(self.sigma_cholesky.diag().prod()) + (self.mu.shape[0] / 2) * np.log(np.pi)
        return -(normalization_factor + 0.5 * torch.sum((x-self.mu) * (torch.potri(self.sigma_cholesky) @ (x-self.mu))))

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.mu

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return torch.diag(self.sigma)