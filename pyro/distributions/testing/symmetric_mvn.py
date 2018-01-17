from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.multivariate_normal import MultivariateNormal
from pyro.distributions.util import copy_docs_from
from torch.autograd import Function, Variable


def _MVN_backward(white, sigma_cholesky, grad_output):
    grad = (grad_output.unsqueeze(-2) * white.unsqueeze(-1)).squeeze(0)
    white_grad = torch.mm(sigma_cholesky.t(), grad)
    white_grad = 0.5 * (white_grad + white_grad.t())
    grad = torch.trtrs(white_grad, sigma_cholesky, transpose=True)[0].t()
    return grad_output, torch.triu(grad)


class _SymmetricSample(Function):
    @staticmethod
    def forward(ctx, mu, sigma_cholesky):
        ctx.save_for_backward(sigma_cholesky)
        ctx.white = mu.new(mu.size()).normal_()
        return mu + torch.mm(ctx.white, sigma_cholesky)

    @staticmethod
    def backward(ctx, grad_output):
        sigma_cholesky, = ctx.saved_variables
        return _MVN_backward(Variable(ctx.white), sigma_cholesky, grad_output)


@copy_docs_from(MultivariateNormal)
class SymmetricMultivariateNormal(MultivariateNormal):
    def sample(self):
        mu = self.mu.expand(self.batch_size, *self.mu.size())
        result = _SymmetricSample.apply(mu, self.sigma_cholesky)
        return result if self.reparameterized else result.detach()
