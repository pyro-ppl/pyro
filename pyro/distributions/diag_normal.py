import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class DiagNormal(Distribution):
    """
    Diagonal covariance Normal
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

    def __init__(self, mu=None, sigma=None, *args, **kwargs):
        """
        Constructor.
        Currently operates over sigma instead of log_sigma - potential problem?
        """
        # if mu sigma no batch dim, add it to mu and sigma
        self.mu = mu
        self.sigma = sigma
        if mu is not None:
            if mu.dim() == 1 and batch_size > 1:
                self.mu = mu.unsqueeze(0).expand(batch_size, mu.size(0))
                self.sigma = sigma.unsqueeze(0).expand(batch_size, sigma.size(0))
        super(DiagNormal, self).__init__(*args, **kwargs)
        self.reparametrized = True

    def sample(self, mu=None, sigma=None, *args, **kwargs):
        """
        Reparametrized diagonal Normal sampler.
        """
        _mu, _sigma = self._sanitize_input(mu, sigma)
        eps = Variable(torch.randn(_mu.size()))
        z = _mu + eps * _sigma
        return z

    def log_pdf(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """
        Diagonal Normal log-likelihood
        """
        _mu, _sigma = self._sanitize_input(mu, sigma)
        log_pxs = -1 * torch.add(torch.add(torch.log(_sigma),
                                 0.5 * torch.log(2.0 * np.pi *
                                 Variable(torch.ones(_sigma.size())))),
                                 0.5 * torch.pow(((x - _mu) / _sigma), 2))
        return torch.sum(log_pxs)

    def batch_log_pdf(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """
        Diagonal Normal log-likelihood
        """
        # expand to patch size of input
        _mu, _sigma = self._sanitize_input(mu, sigma)
        if x.dim() == 1 and _mu.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, _mu, _sigma)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        log_pxs = -1 * torch.add(torch.add(torch.log(_sigma),
                                 0.5 * torch.log(2.0 * np.pi *
                                 Variable(torch.ones(_sigma.size())))),
                                 0.5 * torch.pow(((x - _mu) / _sigma), 2))
        return torch.sum(log_pxs, 1)
