import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class DiagNormal(Distribution):
    """
    Diagonal covariance Normal
    """

    def sanitize_input(self, mu, sigma):
        if mu is not None:
            # stateless distribution
            return mu, sigma
        elif self.mu is not None:
            # stateful distribution
            return self.mu, self.sigma

    # def __init__(self, mu=1, sigma=1, *args, **kwargs):
    #     """
    #     Constructor.
    #     Currently operates over sigma instead of log_sigma - potential problem?
    #     """
    #     # if mu sigma no batch dim, add it to mu and sigma
    #     if mu.dim() == 1 and batch_size > 1:
    #         self.mu = mu.unsqueeze(0).expand(batch_size, mu.size(0))
    #         self.sigma = sigma.unsqueeze(0).expand(batch_size, sigma.size(0))
    #     else:
    #         self.mu = mu
    #         self.sigma = sigma
    #     self.bs = batch_size
    #     super(DiagNormal, self).__init__(*args, **kwargs)
    #     self.reparametrized = True

    def __init__(self, mu=None, sigma=None, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `sigma` - root variance
        """
        self.mu = mu
        self.sigma = sigma
        super(DiagNormal, self).__init__(*args, **kwargs)

    def sample(self, mu, sigma, batch_size=1, *args, **kwargs):
        """
        Reparametrized diagonal Normal sampler.
        """
        _mu, _sigma = self.sanitize_input(mu, sigma)
        eps = Variable(torch.randn(_mu.size()))
        z = _mu + eps * _sigma
        return z

    def log_pdf(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """
        Diagonal Normal log-likelihood
        """
        _mu, _sigma = self.sanitize_input(mu, sigma)
        log_pxs = -1 * torch.add(torch.add(torch.log(_sigma),
                                 0.5 * torch.log(2.0 * np.pi *
                                 Variable(torch.ones(_sigma.size())))),
                                 0.5 * torch.pow(((x - _mu) / _sigma), 2))
        return torch.sum(log_pxs)

<<<<<<< HEAD
    def batch_log_pdf(self, x, mu, sigma, batch_size=1):
        """
        Diagonal Normal log-likelihood
        """
=======
    def batch_log_pdf(self, x, batch_size=1):
>>>>>>> dist-cleanup
        # expand to patch size of input
        _mu, _sigma = self.sanitize_input(mu, sigma)
        if x.dim() == 1 and _mu.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, _mu, _sigma)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        log_pxs = -1 * torch.add(torch.add(torch.log(_sigma),
                                 0.5 * torch.log(2.0 * np.pi *
                                 Variable(torch.ones(_sigma.size())))),
                                 0.5 * torch.pow(((x - _mu) / _sigma), 2))
        return torch.sum(log_pxs, 1)
