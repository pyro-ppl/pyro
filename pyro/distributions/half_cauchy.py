import numbers

import numpy as np
import scipy.stats as spr
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class HalfCauchy(Distribution):
    """
    :param mu: mean *(tensor)*
    :param gamma: scale *(tensor (0, Infinity))*

    Continuous distribution with a positive domain (x > mu).
    """

    def _sanitize_input(self, mu, gamma):
        if mu is not None:
            # stateless distribution
            return mu, gamma
        elif self.mu is not None:
            # stateful distribution
            return self.mu, self.gamma
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, mu=None, gamma=None, batch_size=None, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `gamma` - scale
        """
        self.mu = mu
        self.gamma = gamma
        if mu is not None:
            # this will be deprecated in a future PR
            if mu.dim() == 1 and batch_size is not None:
                self.mu = mu.expand(batch_size, mu.size(0))
                self.gamma = gamma.expand(batch_size, gamma.size(0))
        super(HalfCauchy, self).__init__(*args, **kwargs)

    def batch_shape(self, mu=None, gamma=None, *args, **kwargs):
        mu, gamma = self._sanitize_input(mu, gamma)
        event_dim = 1
        return mu.size()[:-event_dim]

    def event_shape(self, mu=None, gamma=None, *args, **kwargs):
        mu, gamma = self._sanitize_input(mu, gamma)
        event_dim = 1
        return mu.size()[-event_dim:]

    def sample(self, mu=None, gamma=None, *args, **kwargs):
        """
        Half Cauchy sampler.
        """
        mu, gamma = self._sanitize_input(mu, gamma)
        assert mu.dim() == gamma.dim()
        np_sample = spr.halfcauchy.rvs(mu.data.cpu().numpy(), scale=gamma.data.cpu().numpy())
        if isinstance(np_sample, numbers.Number):
            np_sample = [np_sample]
        sample = Variable(torch.Tensor(np_sample).type_as(mu.data))
        return sample

    def batch_log_pdf(self, x, mu=None, gamma=None, *args, **kwargs):
        """
        Half Cauchy log-likelihood
        """
        # expand to patch size of input
        mu, gamma = self._sanitize_input(mu, gamma)
        if x.size() != mu.size():
            mu = mu.expand_as(x)
            gamma = gamma.expand_as(x)
        x_0 = torch.pow((x - mu) / gamma, 2)
        px = 2 / (np.pi * gamma * (1 + x_0))
        batch_log_pdf_shape = self.batch_shape(mu, gamma) + (1,)
        return torch.sum(torch.log(px), -1).contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self, mu=None, gamma=None):
        raise ValueError("Half Cauchy has no defined mean")

    def analytic_var(self,  mu=None, gamma=None):
        raise ValueError("Half Cauchy has no defined variance")
