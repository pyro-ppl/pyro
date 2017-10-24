import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import torch_zeros_like


class Cauchy(Distribution):
    """
    :param mu: mean *(tensor)*
    :param gamma: scale *(tensor (0, Infinity))*

    AKA Lorentz distribution. A continuous distribution which is roughly the ratio of two
    Gaussians if the second Gaussian is zero mean. The distribution is over tensors that
    have the same shape as the parameters ``mu``and ``gamma``, which in turn must have
    the same shape as each other.
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
        super(Cauchy, self).__init__(*args, **kwargs)

    def batch_shape(self, mu=None, gamma=None):
        mu, gamma = self._sanitize_input(mu, gamma)
        event_dim = 1
        return mu.size()[:-event_dim]

    def event_shape(self, mu=None, gamma=None):
        mu, gamma = self._sanitize_input(mu, gamma)
        event_dim = 1
        return mu.size()[-event_dim:]

    def sample(self, mu=None, gamma=None):
        """
        Cauchy sampler.
        """
        mu, gamma = self._sanitize_input(mu, gamma)
        assert mu.dim() == gamma.dim()
        mu_val, gamma_val = mu, gamma
        if mu.dim() > 1:
            # mu and gamma must be size 1 Variables
            mu_val = mu.squeeze()
            gamma_val = gamma.squeeze()
        sample = Variable(torch_zeros_like(mu.data))
        # FIXME: This just fills the entire tensor with the first value
        # Refer to (https://github.com/uber/pyro/issues/302)
        sample.data.cauchy_(mu_val.data[0], gamma_val.data[0])
        return sample

    def batch_log_pdf(self, x, mu=None, gamma=None):
        """
        Cauchy log-likelihood
        """
        # expand to patch size of input
        mu, gamma = self._sanitize_input(mu, gamma)
        x_0 = torch.pow((x - mu)/gamma, 2)
        px = np.pi * gamma * (1 + x_0)
        return -1 * torch.log(px)

    def analytic_mean(self, mu=None, gamma=None):
        raise ValueError("Cauchy has no defined mean")

    def analytic_var(self,  mu=None, gamma=None):
        raise ValueError("Cauchy has no defined variance")
