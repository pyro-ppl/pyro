import numpy as np
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution


class Normal(Distribution):
    """
    Multi-variate normal with arbitrary covariance sigma
    parameterized by its mean mu and covariance matrix sigma
    """

    def sanitize_input(self, mu, sigma):
        if mu is not None:
            # stateless distribution
            return mu, sigma
        elif self.mu is not None:
            # stateful distribution
            return self.mu, self.sigma
        else:
            raise ValueError("Mu and/or sigma had invalid values")

    def __init__(self, mu, sigma, batch_size=1, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `sigma` - root variance
        """
        self.dim = mu.size(0)
        if batch_size == 1 and mu.dim() == 1:
            self.mu = torch.unsqueeze(mu, 1)
        else:
            self.mu = mu.expand(batch_size, mu.size(0))
        self.l_chol = Variable(torch.potrf(sigma.data, False))
        super(Normal, self).__init__(*args, **kwargs)
        self.reparametrized = True

    def sample(self, mu, sigma, *args, **kwargs):
        """
        Reparameterized Normal sampler.
        """
        _mu, _sigma = self.sanitize_input(mu, sigma)
        eps = Variable(torch.randn(_mu.size()))
        if eps.dim() == 1:
            eps = eps.unsqueeze(1)
        z = _mu + torch.mm(self.l_chol, eps).squeeze()
        return z

    def log_pdf(self, x, mu=None, sigma=None, batch_size=1, *args, **kwargs):
        """
        Normal log-likelihood
        """
        _mu, _sigma = self.sanitize_input(mu, sigma)
        ll_1 = Variable(torch.Tensor([-0.5 * self.dim * np.log(2.0 * np.pi)]))
        ll_2 = -torch.sum(torch.log(torch.diag(self.l_chol)))
        x_chol = Variable(
            torch.trtrs(
                (x - _mu).data,
                self.l_chol.data,
                False)[0])
        ll_3 = -0.5 * torch.sum(torch.pow(x_chol, 2.0))

        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, batch_size=1):
        raise NotImplementedError()
