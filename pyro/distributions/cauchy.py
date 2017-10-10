import scipy.stats as spr
import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


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

    def __init__(self, mu=None, gamma=None, batch_size=1, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `gamma` - scale
        """
        self.mu = mu
        self.gamma = gamma
        if mu is not None
	# this will be deprecated in a future PR
            if mu.dim() == 1 and batch_size > 1:
                self.mu = mu.expand(batch_size, mu.size(0))
                self.gamma = gamma.expand(batch_size, gamma.size(0))
        super(Cauchy, self).__init__(*args, **kwargs)

    def sample(self, mu=None, gamma=None, *args, **kwargs):
        """
        Reparameterized diagonal Normal sampler.
        """

        mu, gamma = self._sanitize_input(mu, gamma)
	sample = Variable(torch.Tensor([spr.cauchy.rvs(
            mu, gamma)]).type_as(mu.data))
	return sample

    def log_pdf(self, x, mu=None, gamma=None, batch_size=1, *args, **kwargs):
        """
        Cauchy log-likelihood
        """
        return torch.sum(self.batch_log_pdf(x, mu, gamma, batch_size, *args, **kwargs))

    def batch_log_pdf(self, x, mu=None, gamma=None, batch_size=1, *args, **kwargs):
        """
        Cauchy log-likelihood
        """
        # expand to patch size of input
        mu, gamma = self._sanitize_input(mu, gamma)
        if x.dim() == 1 and mu.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, mu, gamma)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        mu, gamma = self._sanitize_input(mu, gamma)
        x_0 = torch.pow((x - mu)/gamma, 2)
        px = np.pi * gamma * (1 + x_0)
        return -1 * torch.log(px)

    def analytic_mean(self, mu=None, gamma=None):
	raise ValueError("Cauchy has no defined mean".format(type(self)))

    def analytic_var(self,  mu=None, gamma=None):
	raise ValueError("Cauchy has no defined variance".format(type(self)))

