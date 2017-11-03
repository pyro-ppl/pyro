from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Normal(Distribution):
    """
    Univariate normal (Gaussian) distribution.

    A distribution over tensors in which each element is independent and
    Gaussian distributed, with its own mean and standard deviation. The
    distribution is over tensors that have the same shape as the parameters `mu`
    and `sigma`, which in turn must have the same shape as each other.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `sigma` parameters are positive.

    :param torch.autograd.Variable mu: Means.
    :param torch.autograd.Variable sigma: Standard deviations.
        Should be positive and the same shape as `mu`.
    """
    reparameterized = True

    def __init__(self, mu, sigma, batch_size=None, log_pdf_mask=None, *args, **kwargs):
        self.mu = mu
        self.sigma = sigma
        self.log_pdf_mask = log_pdf_mask
        if mu.size() != sigma.size():
            raise ValueError("Expected mu.size() == sigma.size(), but got {} vs {}".format(mu.size(), sigma.size()))
        if mu.dim() == 1 and batch_size is not None:
            self.mu = mu.expand(batch_size, mu.size(0))
            self.sigma = sigma.expand(batch_size, sigma.size(0))
            if log_pdf_mask is not None and log_pdf_mask.dim() == 1:
                self.log_pdf_mask = log_pdf_mask.expand(batch_size, log_pdf_mask.size(0))
        super(Normal, self).__init__(*args, **kwargs)

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
        Reparameterized Normal sampler.

        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        eps = Variable(torch.randn(self.mu.size()).type_as(self.mu.data))
        z = self.mu + eps * self.sigma
        return z if self.reparameterized else z.detach()

    def batch_log_pdf(self, x):
        """
        Diagonal Normal log-likelihood

        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        # expand to patch size of input
        mu = self.mu.expand(self.shape(x))
        sigma = self.sigma.expand(self.shape(x))
        log_pxs = -1 * (torch.log(sigma) + 0.5 * np.log(2.0 * np.pi) + 0.5 * torch.pow((x - mu) / sigma, 2))
        # XXX this allows for the user to mask out certain parts of the score, for example
        # when the data is a ragged tensor. also useful for KL annealing. this entire logic
        # will likely be done in a better/cleaner way in the future
        if self.log_pdf_mask is not None:
            log_pxs = log_pxs * self.log_pdf_mask
        batch_log_pdf = torch.sum(log_pxs, -1)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.mu

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return torch.pow(self.sigma, 2)
