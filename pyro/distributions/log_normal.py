from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


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

    def __init__(self, mu, sigma, batch_size=None, *args, **kwargs):
        self.mu = mu
        self.sigma = sigma
        if mu.size() != sigma.size():
            raise ValueError("Expected mu.size() == sigma.size(), but got {} vs {}".format(mu.size(), sigma.size()))
        if mu.dim() == 1 and batch_size is not None:
            self.mu = mu.expand(batch_size, mu.size(0))
            self.sigma = sigma.expand(batch_size, sigma.size(0))
        super(LogNormal, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        mu = self.mu
        if x is not None:
            if x.size()[-event_dim] != mu.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.mu.size()[-1], but got {} vs {}".format(
                                     x.size(-1), mu.size(-1)))
            try:
                mu = self.mu.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `mu` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(mu.size(), x.size(), str(e)))
        return mu.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        event_dim = 1
        return self.mu.size()[-event_dim:]

    def sample(self):
        """
        Reparameterized log-normal sampler.
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        eps = Variable(torch.randn(self.mu.size()).type_as(self.mu.data))
        z = self.mu + self.sigma * eps
        return torch.exp(z)

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        mu = self.mu.expand(self.shape(x))
        sigma = self.sigma.expand(self.shape(x))
        ll_1 = Variable(torch.Tensor([-0.5 * np.log(2.0 * np.pi)]).type_as(mu.data).expand_as(x))
        ll_2 = -torch.log(sigma * x)
        ll_3 = -0.5 * torch.pow((torch.log(x) - mu) / sigma, 2.0)
        batch_log_pdf = torch.sum(ll_1 + ll_2 + ll_3, -1)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return torch.exp(self.mu + 0.5 * torch.pow(self.sigma, 2.0))

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return (torch.exp(torch.pow(self.sigma, 2.0)) - Variable(torch.ones(1))) * \
            torch.pow(self.analytic_mean(), 2)
