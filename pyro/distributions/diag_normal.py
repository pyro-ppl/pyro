import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class DiagNormal(Distribution):
    """
    Diagonal covariance Normal - the first distribution
    """

    def __init__(self, mu, sigma, batch_size=1, *args, **kwargs):
        """
        Constructor.
        Currently operates over sigma instead of log_sigma - potential problem?
        """
        # if mu sigma no batch dim, add it to mu and sigma
        if mu.dim() == 1 and batch_size > 1:
            self.mu = mu.unsqueeze(0).expand(batch_size, mu.size(0))
            self.sigma = sigma.unsqueeze(0).expand(batch_size, sigma.size(0))
        else:
            self.mu = mu
            self.sigma = sigma
        self.bs = batch_size
        super(DiagNormal, self).__init__(*args, **kwargs)
        self.reparametrized = True

    def sample(self, batch_size=1):
        """
        Reparametrized diagonal Normal sampler.
        """
        if batch_size != 1 and batch_size != self.bs:
            raise ValueError("Batch sizes do not match")

        eps = Variable(torch.randn(self.mu.size()))
        z = self.mu + eps * self.sigma
        return z

    def log_pdf(self, x):
        """
        Diagonal Normal log-likelihood
        """
        log_pxs = -1 * torch.add(torch.add(torch.log(self.sigma),
                                 0.5 * torch.log(2.0 * np.pi *
                                 Variable(torch.ones(self.sigma.size())))),
                                 0.5 * torch.pow(((x - self.mu) / self.sigma), 2))
        return torch.sum(log_pxs)

    def batch_log_pdf(self, x, batch_size=1):
        """
        Diagonal Normal log-likelihood
        """
        # expand to patch size of input
        if x.dim() == 1 and self.mu.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        log_pxs = -1 * torch.add(torch.add(torch.log(self.sigma),
                                 0.5 * torch.log(2.0 * np.pi *
                                 Variable(torch.ones(self.sigma.size())))),
                                 0.5 * torch.pow(((x - self.mu) / self.sigma), 2))
        return torch.sum(log_pxs, 1)
