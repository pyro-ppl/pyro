import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Normal(Distribution):
    """
    :param mu: mean *(real)*
    :param sigma: standard deviation *(real (0, Infinity))*
    :param dims: dimension of tensor *(int (>=1) array)*

    Gaussian Distribution over a tensor of independent variables.
    """
    reparameterized = True

    def _sanitize_input(self, mu, sigma):
        if mu is not None:
            # stateless distribution
            return mu, sigma
        elif self.mu is not None:
            # stateful distribution
            return self.mu, self.sigma
        else:
            raise ValueError("Mu and/or sigma had invalid values")

    def __init__(self, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `sigma` - root variance
        """
        if mu is not None:
            if batch_size == 1 and mu.dim() == 1:
                self.mu = torch.unsqueeze(mu, 1)
            else:
                self.mu = mu.expand(batch_size, mu.size(0))
        super(Normal, self).__init__(*args, **kwargs)

    def sample(self, mu=None, sigma=None, *args, **kwargs):
        """
        Reparameterized Normal sampler.
        """
        mu, sigma = self._sanitize_input(mu, sigma)
        l_chol = Variable(torch.potrf(sigma.data, False).type_as(mu.data))
        eps = Variable(torch.randn(mu.size()).type_as(mu.data))
        if eps.dim() == 1:
            eps = eps.unsqueeze(1)
        z = mu + torch.mm(l_chol, eps).squeeze()
        return z

    def log_pdf(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """
        Normal log-likelihood
        """
        mu, sigma = self._sanitize_input(mu, sigma)
        assert mu.dim() == sigma.dim()
        if x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        if mu.size() != x.size():
            mu = mu.expand_as(x)
            sigma = sigma.expand_as(x)
        l_chol = Variable(torch.potrf(sigma.data, False).type_as(mu.data))
        ll_1 = Variable(torch.Tensor([-0.5 * mu.size(0) * np.log(2.0 * np.pi)])
                        .type_as(mu.data))
        ll_2 = -torch.sum(torch.log(torch.diag(l_chol)))
        x_chol = Variable(
            torch.trtrs(
                (x - mu).data,
                l_chol.data,
                False)[0])
        ll_3 = -0.5 * torch.sum(torch.pow(x_chol, 2.0))

        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, batch_size=1):
        raise NotImplementedError()
