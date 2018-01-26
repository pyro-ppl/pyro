from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(Distribution)
class LogNormal(Distribution):
    """
    Log-normal distribution.

    A distribution over probability vectors obtained by exp-transforming a random
    variable drawn from ``Normal({mu: mu, sigma: sigma})``.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `sigma` parameters are positive.

    :param torch.autograd.Variable mu: log mean parameter.
    :param torch.autograd.Variable sigma: log standard deviations.
        Should be positive.
    """
    reparameterized = True

    def __init__(self, mu, sigma, *args, **kwargs):
        self.mu = mu
        self.sigma = sigma
        if mu.size() != sigma.size():
            raise ValueError("Expected mu.size() == sigma.size(), but got {} vs {}".format(mu.size(), sigma.size()))
        super(LogNormal, self).__init__(*args, **kwargs)

    def batch_shape(self):
        return self.mu.size()

    def event_shape(self):
        return torch.Size()

    def sample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.mu.size()
        eps = Variable(torch.randn(shape).type_as(self.mu.data))
        z = self.mu + self.sigma * eps
        return torch.exp(z)

    def log_prob(self, x):
        ll_1 = Variable(torch.Tensor([-0.5 * np.log(2.0 * np.pi)]).type_as(self.mu.data).expand_as(x))
        ll_2 = -torch.log(self.sigma * x)
        ll_3 = -0.5 * torch.pow((torch.log(x) - self.mu) / self.sigma, 2.0)
        return ll_1 + ll_2 + ll_3

    def analytic_mean(self):
        return torch.exp(self.mu + 0.5 * torch.pow(self.sigma, 2.0))

    def analytic_var(self):
        return (torch.exp(torch.pow(self.sigma, 2.0)) - Variable(torch.ones(1))) * \
            torch.pow(self.analytic_mean(), 2)
