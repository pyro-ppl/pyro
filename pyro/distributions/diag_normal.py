import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class DiagNormal(Distribution):
    """
    :param torch.autograd.Variable mu: mean *(tensor)*
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

    def _expand_dims(self, x, mu, sigma):
        """
        Expand to 2-dimensional tensors of the same shape.
        """
        if not isinstance(x, (torch.Tensor, Variable)):
            raise TypeError('Expected x a Tensor or Variable, got a {}'.format(type(x)))
        if not isinstance(mu, Variable):
            raise TypeError('Expected mu a Variable, got a {}'.format(type(mu)))
        if not isinstance(sigma, Variable):
            raise TypeError('Expected sigma a Variable, got a {}'.format(type(sigma)))

        if x.dim() not in (1, 2):
            raise ValueError('Expected x.dim() in (1,2), actual: {}'.format(x.dim()))
        if mu.dim() not in (1, 2):
            raise ValueError('Expected mu.dim() in (1,2), actual: {}'.format(mu.dim()))
        if mu.size() != sigma.size():
            raise ValueError('expected mu.size() == sigma.size(), actual {} vs {}'.format(
                mu.size(), sigma.size()))
        if x.dim() == 2 and mu.dim() == 2 and x.size(0) != mu.size(0):
            # Disallow broadcasting, e.g. disallow resizing (1,4) -> (4,4).
            raise ValueError('Batch sizes disagree: {} vs {}'.format(x.size(0), mu.size(0)))

        if x.dim() == 1:
            x = x.unsqueeze(0)
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
        batch_size = max(x.size(0), mu.size(0))
        x = x.expand(batch_size, x.size(1))
        mu = mu.expand(batch_size, mu.size(1))
        sigma = sigma.expand(batch_size, mu.size(1))
        return x, mu, sigma

    def __init__(self, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """
        :param mu: A vector of mean parameters.
        :type mu: torch.autograd.Variable or None
        :param sigma: A vector of scale parameters.
        :type sigma: torch.autograd.Variable or None
        :param int batch_size: DEPRECATED.
        """
        assert batch_size == 1, 'DEPRECATED'
        if (mu is None) ^ (sigma is None):
            raise ValueError('mu and sigma must be either both specified or both unspecified')
        self.mu = mu
        self.sigma = sigma
        super(DiagNormal, self).__init__(*args, **kwargs)
        self.reparameterized = True

    def sample(self, mu=None, sigma=None, *args, **kwargs):
        """
        Draws either a single sample (if mu.dim() == 1), or one sample per param (if mu.dim() == 2).

        (Reparameterized).

        :param mu: A vector of mean parameters.
        :type mu: torch.autograd.Variable or None
        :param sigma: A vector of scale parameters.
        :type sigma: torch.autograd.Variable or None
        """
        mu, sigma = self._sanitize_input(mu, sigma)
        eps = Variable(torch.randn(mu.size()).type_as(mu.data))
        z = mu + eps * sigma
        if 'reparameterized' in kwargs:
            self.reparameterized = kwargs['reparameterized']
        if not self.reparameterized:
            return Variable(z.data)
        return z

    def _log_pdf_impl(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        assert batch_size == 1, 'DEPRECATED'
        mu, sigma = self._sanitize_input(mu, sigma)
        x, mu, sigma = self._expand_dims(x, mu, sigma)
        log_pxs = -1 * torch.add(torch.add(torch.log(sigma),
                                 0.5 * torch.log(2.0 * np.pi *
                                 Variable(torch.ones(sigma.size()).type_as(mu.data)))),
                                 0.5 * torch.pow(((x - mu) / sigma), 2))
        return log_pxs

    def log_pdf(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """"
        Evaluates total log probabity density of one or a batch of samples.

        :param torch.autograd.Variable x: A value (if x.dim() == 1) or or batch of values (if x.dim() == 2).
        :param mu: A vector of mean parameters.
        :type mu: torch.autograd.Variable or None.
        :param sigma: A vector of scale parameters.
        :type sigma: torch.autograd.Variable or None.
        :param int batch_size: DEPRECATED.
        :return: log probability densities of each element in the batch.
        :rtype: torch.autograd.Variable of dimension 1.
        """
        log_pxs = self._log_pdf_impl(x, mu, sigma, batch_size, *args, **kwargs)
        if 'log_pdf_mask' in kwargs:
            return torch.sum(kwargs['log_pdf_mask'] * log_pxs)
        return torch.sum(log_pxs)

    def batch_log_pdf(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """"
        Evaluates log probabity density over one or a batch of samples.

        :param torch.autograd.Variable x: A value (if x.dim() == 1) or or batch of values (if x.dim() == 2).
        :param mu: A vector of mean parameters.
        :type mu: torch.autograd.Variable or None.
        :param sigma: A vector of scale parameters.
        :type sigma: torch.autograd.Variable or None.
        :param int batch_size: DEPRECATED.
        :return: log probability densities of each element in the batch.
        :rtype: torch.autograd.Variable of dimension 1.
        """
        log_pxs = self._log_pdf_impl(x, mu, sigma, batch_size, *args, **kwargs)
        return torch.sum(log_pxs, 1)

    def analytic_mean(self, mu=None, sigma=None):
        mu, sigma = self._sanitize_input(mu, sigma)
        return mu

    def analytic_var(self,  mu=None, sigma=None):
        mu, sigma = self._sanitize_input(mu, sigma)
        return torch.pow(sigma, 2)
