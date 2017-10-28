import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Normal(Distribution):
    """
    :param mu: mean *(tensor)*
    :param sigma: standard deviations *(tensor (0, Infinity))*

    A distribution over tensors in which each element is independent and
    Gaussian distributed, with its own mean and standard deviation. i.e. A
    multivariate Gaussian distribution with diagonal covariance matrix. The
    distribution is over tensors that have the same shape as the parameters ``mu``
    and ``sigma``, which in turn must have the same shape as each other.
    """
    reparameterized = True

    def __init__(self, mu, sigma, batch_size=None, log_pdf_mask=None, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `sigma` - root variance
        """
        self.mu = mu
        self.sigma = sigma
        self.log_pdf_mask = log_pdf_mask
        if mu.size() != sigma.size():
            raise ValueError("Expected mu.size() == sigma.size(), but got {} vs {}"
                             .format(mu.size(), sigma.size()))
        if mu.dim() == 1 and batch_size is not None:
            self.mu = mu.expand(batch_size, mu.size(0))
            self.sigma = sigma.expand(batch_size, sigma.size(0))
            if log_pdf_mask is not None and log_pdf_mask.dim() == 1:
                self.log_pdf_mask = log_pdf_mask.expand(batch_size, log_pdf_mask.size(0))
        super(Normal, self).__init__(*args, **kwargs)

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
        Reparameterized diagonal Normal sampler.
        """
        eps = Variable(torch.randn(self.mu.size()).type_as(self.mu.data))
        z = self.mu + eps * self.sigma
        return z if self.reparameterized else z.detach()

    def batch_log_pdf(self, x):
        """
        Diagonal Normal log-likelihood
        """
        # expand to patch size of input
        mu = self.mu.expand(self.shape(x))
        sigma = self.sigma.expand(self.shape(x))
        log_pxs = -1 * torch.add(torch.add(torch.log(sigma),
                                 0.5 * torch.log(2.0 * np.pi *
                                 Variable(torch.ones(sigma.size()).type_as(mu.data)))),
                                 0.5 * torch.pow(((x - mu) / sigma), 2))
        # XXX this allows for the user to mask out certain parts of the score, for example
        # when the data is a ragged tensor. also useful for KL annealing. this entire logic
        # will likely be done in a better/cleaner way in the future
        if self.log_pdf_mask is not None:
            log_pxs = log_pxs * self.log_pdf_mask
        batch_log_pdf = torch.sum(log_pxs, -1)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        return self.mu

    def analytic_var(self):
        return torch.pow(self.sigma, 2)
