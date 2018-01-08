from __future__ import absolute_import, division, print_function

from unittest import TestCase

import torch
from torch.autograd import Variable, grad
from pyro.distributions import MultivariateNormal
from tests.common import assert_equal


class TestMultivariateNormal(TestCase):
    """
    Tests if the gradients of batch_log_pdf are the same regardless of normalization. The test is run once for a
    distribution that is parameterized by the full covariance matrix and once for one that is parameterized by the
    cholesky decomposition of the covariance matrix.
    """

    def setUp(self):
        N = 400
        self.L_tensor = torch.tril(1e-3 * torch.ones(N, N)).t()
        self.mu = Variable(torch.rand(N))
        self.L = Variable(self.L_tensor, requires_grad=True)
        self.sigma = Variable(torch.mm(self.L_tensor.t(), self.L_tensor), requires_grad=True)
        # Draw from an unrelated distribution as not to interfere with the gradients
        self.sample = Variable(torch.randn(N))

        self.cholesky_mv_normalized = MultivariateNormal(self.mu, scale_tril=self.L, normalized=True)
        self.cholesky_mv = MultivariateNormal(self.mu, scale_tril=self.L, normalized=False)

        self.full_mv_normalized = MultivariateNormal(self.mu, self.sigma, normalized=True)
        self.full_mv = MultivariateNormal(self.mu, self.sigma, normalized=False)

    def test_log_pdf_gradients_cholesky(self):
        grad1 = grad([self.cholesky_mv.log_pdf(self.sample)], [self.L])[0].data
        grad2 = grad([self.cholesky_mv_normalized.log_pdf(self.sample)], [self.L])[0].data
        assert_equal(grad1, grad2)

    def test_log_pdf_gradients(self):
        grad1 = grad([self.full_mv.log_pdf(self.sample)], [self.sigma])[0].data
        grad2 = grad([self.full_mv_normalized.log_pdf(self.sample)], [self.sigma])[0].data
        assert_equal(grad1, grad2)
