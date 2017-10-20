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

    def __init__(self, mu=None, gamma=None, batch_size=1, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `gamma` - scale
        """
        self.mu = mu
        self.gamma = gamma
        if mu is not None:
            # this will be deprecated in a future PR
            if mu.dim() == 1 and batch_size > 1:
                self.mu = mu.expand(batch_size, mu.size(0))
                self.gamma = gamma.expand(batch_size, gamma.size(0))
        super(HalfCauchy, self).__init__(*args, **kwargs)

    def sample(self, mu=None, gamma=None, *args, **kwargs):
        """
        Half Cauchy sampler.
        """
        mu, gamma = self._sanitize_input(mu, gamma)
        assert mu.dim() == gamma.dim()
        if mu.dim() > 1:
            # mu and gamma must be size 1 Variables
            mu = mu.squeeze()
            gamma = gamma.squeeze()
        sample = Variable(torch.Tensor([spr.halfcauchy.rvs(
                     mu.data.numpy(), scale=gamma.data.numpy())])
                     .type_as(mu.data))
        return sample

    def batch_log_pdf(self, x, mu=None, gamma=None, batch_size=1, *args, **kwargs):
        """
        Half Cauchy log-likelihood
        """
        # expand to patch size of input
        mu, gamma = self._sanitize_input(mu, gamma)
        if x.dim() != mu.dim():
            x = x.expand(batch_size, x.size(0))
        x_0 = torch.pow((x - mu) / gamma, 2)
        px = 2 / (np.pi * gamma * (1 + x_0))
        return torch.log(px)

    def analytic_mean(self, mu=None, gamma=None):
        raise ValueError("Half Cauchy has no defined mean")

    def analytic_var(self,  mu=None, gamma=None):
        raise ValueError("Half Cauchy has no defined variance")
