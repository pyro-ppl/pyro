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

    def _sanitize_input(self, mu, sigma):
        if mu is not None:
            # stateless distribution
            return mu, sigma
        elif self.mu is not None:
            # stateful distribution
            return self.mu, self.sigma
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, mu=None, sigma=None, batch_size=1, *args, **kwargs):
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
            elif mu.dim() == 1:
                self.mu = mu.expand(batch_size, mu.size(0))
                self.sigma = sigma.expand(batch_size, sigma.size(0))
        super(LogNormal, self).__init__(*args, **kwargs)
        self.reparameterized = True

    def sample(self, mu=None, sigma=None, *args, **kwargs):
        """
        Reparameterized log-normal sampler.
        """
        _mu, _sigma = self._sanitize_input(mu, sigma)
        eps = Variable(torch.randn(1).type_as(_mu.data))
        z = _mu + _sigma * eps
        return torch.exp(z)

    def log_pdf(self, x, mu=None, sigma=None, *args, **kwargs):
        """
        log-normal log-likelihood
        """
        _mu, _sigma = self._sanitize_input(mu, sigma)
        ll_1 = Variable(torch.Tensor([-0.5 * np.log(2.0 * np.pi)]).type_as(_mu.data))
        ll_2 = -torch.log(_sigma * x)
        ll_3 = -0.5 * torch.pow((torch.log(x) - _mu) / _sigma, 2.0)
        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """
        log-normal log-likelihood
        """
        _mu, _sigma = self._sanitize_input(mu, sigma)
        if x.dim() == 1 and _mu.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, _mu, _sigma)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll_1 = Variable(torch.Tensor([-0.5 * np.log(2.0 * np.pi)])
                        .type_as(_mu.data).expand_as(x))
        ll_2 = -torch.log(_sigma * x)
        ll_3 = -0.5 * torch.pow((torch.log(x) - _mu) / _sigma, 2.0)
        return ll_1 + ll_2 + ll_3
