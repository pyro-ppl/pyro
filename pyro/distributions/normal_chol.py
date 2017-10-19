import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class NormalChol(Distribution):
    """
    :param mu: mean *(real)*
    :param sigma: standard deviation *(real (0, Infinity))*
    :param L: Cholesky decomposition

    A multi-variate normal distribution with arbitrary covariance sigma
    parameterized by its mean and its cholesky decomposition ``L``. Parameters
    must have dimensions <= 2.
    """
    reparameterized = False  # This is treated as non-reparameterized because chol does not support autograd.

    def _sanitize_input(self, mu, sigma):
        if mu is not None:
            # stateless distribution
            return mu, sigma
        elif self.mu is not None:
            # stateful distribution
            return self.mu, self.L
        else:
            raise ValueError("Mu and/or sigma had invalid values")

    def __init__(self, mu=None, L=None, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `L` - cholesky decomposition matrix
        """
        self.mu = mu
        self.L = L
        super(NormalChol, self).__init__(*args, **kwargs)

    def sample(self, mu=None, L=None, *args, **kwargs):
        """
        Reparameterized Normal cholesky sampler.
        """
        mu, L = self._sanitize_input(mu, L)
        eps = Variable(torch.randn(mu.size()).type_as(mu.data))
        if eps.dim() == 1:
            eps = eps.unsqueeze(1)
        z = mu + torch.mm(L, eps).squeeze()
        return z

    def log_pdf(self, x, mu=None, L=None, *args, **kwargs):
        """
        Normal cholesky log-likelihood
        """
        mu, L = self._sanitize_input(mu, L)
        ll_1 = Variable(torch.Tensor([-0.5 * mu.size(0) * np.log(2.0 * np.pi)])
                        .type_as(mu.data))
        ll_2 = -torch.sum(torch.log(torch.diag(L)))
        x_chol = Variable(
            torch.trtrs(
                (x - mu).unsqueeze(1).data,
                L.data,
                False)[0])
        ll_3 = -0.5 * torch.sum(torch.pow(x_chol, 2.0))

        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, mu=None, L=None, batch_size=1, *args, **kwargs):
        raise NotImplementedError()

    def analytic_mean(self, mu=None, L=None):
        mu, L = self._sanitize_input(mu, L)
        return mu

    def analytic_var(self,  mu=None, L=None):
        mu, L = self._sanitize_input(mu, L)
        cov = torch.mm(L, torch.transpose(L, 0, 1))
        return torch.diag(cov)
