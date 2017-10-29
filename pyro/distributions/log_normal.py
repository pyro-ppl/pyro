import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class LogNormal(Distribution):
    """
    :param mu: mean *(vector)*
    :param sigma: standard deviations *(vector (0, Infinity))*

    A distribution over probability vectors obtained by exp-transforming a random
    variable drawn from ``Normal({mu: mu, sigma: sigma})``.
    """
    reparameterized = True

    def __init__(self, mu, sigma, batch_size=None, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `sigma` - root variance
        """
        self.mu = mu
        self.sigma = sigma
        if mu.size() != sigma.size():
            raise ValueError("Expected mu.size() == sigma.size(), but got {} vs {}"
                             .format(mu.size(), sigma.size()))
        if mu.dim() == 1 and batch_size is not None:
            self.mu = mu.expand(batch_size, mu.size(0))
            self.sigma = sigma.expand(batch_size, sigma.size(0))
        super(LogNormal, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        event_dim = 1
        mu = self.mu
        if x is not None and x.size() != mu.size():
            mu = self.mu.expand(x.size()[:-event_dim] + self.event_shape())
        return mu.size()[:-event_dim]

    def event_shape(self):
        event_dim = 1
        return self.mu.size()[-event_dim:]

    def shape(self, x=None):
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        """
        Reparameterized log-normal sampler.
        """
        eps = Variable(torch.randn(self.mu.size()).type_as(self.mu.data))
        z = self.mu + self.sigma * eps
        return torch.exp(z)

    def batch_log_pdf(self, x):
        """
        log-normal log-likelihood
        """
        mu = self.mu.expand(self.shape(x))
        sigma = self.sigma.expand(self.shape(x))
        ll_1 = Variable(torch.Tensor([-0.5 * np.log(2.0 * np.pi)])
                        .type_as(mu.data).expand_as(x))
        ll_2 = -torch.log(sigma * x)
        ll_3 = -0.5 * torch.pow((torch.log(x) - mu) / sigma, 2.0)
        batch_log_pdf = torch.sum(ll_1 + ll_2 + ll_3, -1)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        return torch.exp(self.mu + 0.5 * torch.pow(self.sigma, 2.0))

    def analytic_var(self):
        return (torch.exp(torch.pow(self.sigma, 2.0)) - Variable(torch.ones(1))) * \
            torch.pow(self.analytic_mean(), 2)
