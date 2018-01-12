from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.multivariate_normal import MultivariateNormal
from pyro.distributions.util import copy_docs_from
from torch.autograd import Function


class _SymmetricSample(Function):
    def forward(self, ctx, mu, sigma_cholesky):
        self.white = mu.new(mu.size()).normal_()
        return mu + torch.mm(self.white, sigma_cholesky)

    def backward(self, ctx, grad_output):
        asymmetric = grad_output.unsqueeze(-1) * self.white.unsqueeze(-2)
        grad = asymmetric + asymmetric.transpose(-1, -2)
        grad *= 0.5
        return grad


@copy_docs_from(MultivariateNormal)
class SymmetricMultivariateNormal(MultivariateNormal):
    def sample(self):
        mu = self.mu.expand(self.batch_size, self.mu.size())
        result = _SymmetricSample(mu, self.sigma_cholesky)
        return result if self.reparameterized else result.detach()
