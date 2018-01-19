from __future__ import absolute_import, division, print_function

import math

import numpy as np

import torch
from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from
from torch.autograd import Function, Variable


@copy_docs_from(Distribution)
class BivariateNormal(Distribution):
    reparameterized = True

    def __init__(self, loc, scale_triu, batch_size=None):
        self.loc = loc
        self.scale_triu = scale_triu
        self.batch_size = 1 if batch_size is None else batch_size

    def batch_shape(self, x=None):
        loc = self.loc.expand(self.batch_size, *self.loc.size()).squeeze(0)
        if x is not None:
            if x.size()[-1] != loc.size()[-1]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.loc.size()[0], but got {} vs {}".format(
                                     x.size(-1), loc.size(-1)))
            try:
                loc = loc.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `loc` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(loc.size(), x.size(), str(e)))

        return loc.size()[:-1]

    def event_shape(self):
        return self.loc.size()[-1:]

    def sample(self):
        return self.loc + torch.mv(self.scale_triu.t(), self.loc.new(self.loc.shape).normal_(), )

    def batch_log_pdf(self, x):
        delta = x - self.loc
        z0 = delta[..., 0] / self.scale_triu[..., 0, 0]
        z1 = (delta[..., 1] - self.scale_triu[..., 0, 1] * z0) / self.scale_triu[..., 1, 1]
        z = torch.stack([z0, z1], dim=-1)
        mahalanobis_squared = (z ** 2).sum(-1)
        normalization_constant = self.scale_triu.diag().log().sum(-1) + np.log(2 * np.pi)
        return -(normalization_constant + 0.5 * mahalanobis_squared).unsqueeze(-1)

    def entropy(self):
        return self.scale_triu.diag().log().sum() + (1 + math.log(2 * math.pi))


def _BVN_backward_fritz(white, scale_triu, grad_output):
    grad = (grad_output.unsqueeze(-2) * white.unsqueeze(-1)).squeeze(0)
    white_grad = torch.mm(scale_triu.t(), grad)
    white_grad = 0.5 * (white_grad + white_grad.t())
    grad = torch.trtrs(white_grad, scale_triu, transpose=True)[0].t()
    return grad_output, torch.triu(grad)


# TODO get this to work
def _BVN_backward_martin(white, scale_triu, grad_output):
    grad = (grad_output.unsqueeze(-2) * white.unsqueeze(-1)).squeeze(0)
    x = torch.trtrs(white, scale_triu, transpose=True)[0]
    y = torch.mm(scale_triu.t(), grad_output)
    grad += (x.unsqueeze(-2) * y.unsqueeze(-1)).squeeze(0)
    grad *= 0.5
    return grad_output, torch.triu(grad)


_BVN_backward = _BVN_backward_fritz


class _SymmetricSample(Function):
    @staticmethod
    def forward(ctx, loc, scale_triu):
        ctx.save_for_backward(scale_triu)
        ctx.white = loc.new(loc.size()).normal_()
        return loc + torch.mm(ctx.white, scale_triu)

    @staticmethod
    def backward(ctx, grad_output):
        scale_triu, = ctx.saved_variables
        return _BVN_backward(Variable(ctx.white), scale_triu, grad_output)


@copy_docs_from(BivariateNormal)
class SymmetricBivariateNormal(BivariateNormal):
    def sample(self):
        loc = self.loc.expand(self.batch_size, *self.loc.size())
        return _SymmetricSample.apply(loc, self.scale_triu)
