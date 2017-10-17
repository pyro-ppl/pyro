import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class DiagNormal(Distribution):
    """
    :param mu: mean *(tensor)*
    :param sigma: standard deviations *(tensor (0, Infinity))*

    A distribution over tensors in which each element is independent and
    Gaussian distributed, with its own mean and standard deviation. i.e. A
    multivariate Gaussian distribution with diagonal covariance matrix. The
    distribution is over tensors that have the same shape as the parameters ``mu``
    and ``sigma``, which in turn must have the same shape as each other.
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
            if mu.dim() == 1 and batch_size > 1:
                self.mu = mu.expand(batch_size, mu.size(0))
                self.sigma = sigma.expand(batch_size, sigma.size(0))
        super(DiagNormal, self).__init__(*args, **kwargs)
        self.reparameterized = True

    def sample(self, mu=None, sigma=None, *args, **kwargs):
        """
        Reparameterized diagonal Normal sampler.
        """
        mu, sigma = self._sanitize_input(mu, sigma)
        eps = Variable(torch.randn(mu.size()).type_as(mu.data))
        z = mu + eps * sigma
        if 'reparameterized' in kwargs:
            self.reparameterized = kwargs['reparameterized']
        if not self.reparameterized:
            return Variable(z.data)
        return z

    def batch_log_pdf(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """
        Diagonal Normal log-likelihood
        """
        # expand to patch size of input
        mu, sigma = self._sanitize_input(mu, sigma)
        assert mu.dim() == sigma.dim()
        if x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        if mu.size() != x.size():
            mu = mu.expand_as(x)
            sigma = sigma.expand_as(x)
        log_pxs = -1 * torch.add(torch.add(torch.log(sigma),
                                 0.5 * torch.log(2.0 * np.pi *
                                 Variable(torch.ones(sigma.size()).type_as(mu.data)))),
                                 0.5 * torch.pow(((x - mu) / sigma), 2))
        batch_probs = torch.sum(log_pxs, 1)
        if 'log_pdf_mask' in kwargs:
            mask = kwargs['log_pdf_mask']
            if mask is not None:
                batch_probs = batch_probs * mask.view(-1)
        return batch_probs

    def analytic_mean(self, mu=None, sigma=None):
        mu, sigma = self._sanitize_input(mu, sigma)
        return mu

    def analytic_var(self,  mu=None, sigma=None):
        mu, sigma = self._sanitize_input(mu, sigma)
        return torch.pow(sigma, 2)
