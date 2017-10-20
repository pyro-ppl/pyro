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

    def _sanitize_input(self, mu, sigma):
        if mu is not None:
            # stateless distribution
            return mu, sigma
        elif self.mu is not None:
            # stateful distribution
            return self.mu, self.sigma
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, mu=None, sigma=None, batch_size=None, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `sigma` - root variance
        """
        self.mu = mu
        self.sigma = sigma
        if mu is not None:
            if mu.dim() != sigma.dim():
                raise ValueError("Mu and sigma need to have the same dimensions.")
            elif mu.dim() == 1 and batch_size is not None:
                self.mu = mu.expand(batch_size, mu.size(0))
                self.sigma = sigma.expand(batch_size, sigma.size(0))
        super(LogNormal, self).__init__(*args, **kwargs)

    def batch_shape(self, mu=None, sigma=None, *args, **kwargs):
        mu, sigma = self._sanitize_input(mu, sigma)
        event_dim = 1
        return mu.size()[:-event_dim]

    def event_shape(self, mu=None, sigma=None, *args, **kwargs):
        mu, sigma = self._sanitize_input(mu, sigma)
        event_dim = 1
        return mu.size()[-event_dim:]

    def sample(self, mu=None, sigma=None, *args, **kwargs):
        """
        Reparameterized log-normal sampler.
        """
        mu, sigma = self._sanitize_input(mu, sigma)
        eps = Variable(torch.randn(1).type_as(mu.data))
        z = mu + sigma * eps
        return torch.exp(z)

    def batch_log_pdf(self, x, mu=None, sigma=None, *args, **kwargs):
        """
        log-normal log-likelihood
        """
        mu, sigma = self._sanitize_input(mu, sigma)
        assert mu.dim() == sigma.dim()
        if mu.size() != sigma.size():
            mu = mu.expand_as(x)
            sigma = sigma.expand_as(x)
        ll_1 = Variable(torch.Tensor([-0.5 * np.log(2.0 * np.pi)])
                        .type_as(mu.data).expand_as(x))
        ll_2 = -torch.log(sigma * x)
        ll_3 = -0.5 * torch.pow((torch.log(x) - mu) / sigma, 2.0)
        log_pdf = torch.sum(ll_1 + ll_2 + ll_3, -1)
        batch_log_pdf_shape = x.size()[:-1] + (1,)
        return log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self, mu=None, sigma=None):
        mu, sigma = self._sanitize_input(mu, sigma)
        return torch.exp(mu + 0.5 * torch.pow(sigma, 2.0))

    def analytic_var(self, mu=None, sigma=None):
        mu, sigma = self._sanitize_input(mu, sigma)
        return (torch.exp(torch.pow(sigma, 2.0)) - Variable(torch.ones(1))) * \
            torch.pow(self.analytic_mean(mu, sigma), 2)
