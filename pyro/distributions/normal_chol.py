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
        self.reparameterized = False

    def sample(self, mu=None, L=None, *args, **kwargs):
        """
        Reparameterized Normal cholesky sampler.
        """
        _mu, _L = self._sanitize_input(mu, L)
        eps = Variable(torch.randn(_mu.size()).type_as(_mu.data))
        if eps.dim() == 1:
            eps = eps.unsqueeze(1)
        z = _mu + torch.mm(_L, eps).squeeze()
        return z

    def log_pdf(self, x, mu=None, L=None, *args, **kwargs):
        """
        Normal cholesky log-likelihood
        """
        _mu, _L = self._sanitize_input(mu, L)
        ll_1 = Variable(torch.Tensor([-0.5 * _mu.size(0) * np.log(2.0 * np.pi)])
                        .type_as(_mu.data))
        ll_2 = -torch.sum(torch.log(torch.diag(_L)))
        x_chol = Variable(
            torch.trtrs(
                (x - _mu).unsqueeze(1).data,
                _L.data,
                False)[0])
        ll_3 = -0.5 * torch.sum(torch.pow(x_chol, 2.0))

        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, batch_size=1):
        raise NotImplementedError()
