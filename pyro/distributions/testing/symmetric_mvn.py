from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.multivariate_normal import MultivariateNormal
from pyro.distributions.util import copy_docs_from
from torch.autograd import Function, Variable


class _SymmetricSample(Function):
    @staticmethod
    def forward(ctx, mu, sigma_cholesky):
        ctx.white = mu.new(mu.size()).normal_()
        return mu + torch.mm(ctx.white, sigma_cholesky)

    @staticmethod
    def backward(ctx, grad_output):
        grad = (grad_output.unsqueeze(-2) * Variable(ctx.white.unsqueeze(-1))).squeeze(0)
        grad = grad + grad.transpose(-1, -2)
        grad *= 0.5
        return grad_output, torch.triu(grad)


@copy_docs_from(MultivariateNormal)
class SymmetricMultivariateNormal(MultivariateNormal):
    def sample(self):
        mu = self.mu.expand(self.batch_size, *self.mu.size())
        result = _SymmetricSample.apply(mu, self.sigma_cholesky)
        return result if self.reparameterized else result.detach()
