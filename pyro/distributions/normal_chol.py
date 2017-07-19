import numpy as np
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution


class NormalChol(Distribution):
    """
    Multi-variate normal with arbitrary covariance sigma
    parameterized by its mean and its cholesky decomposition L
    """

    def __init__(self, mu, L, *args, **kwargs):
        """
        Params:
          `mu` - mean
          `L` - cholesky decomposition matrix
        """
        self.mu = mu
        self.L = L
        self.dim = mu.size(0)
        super(NormalChol, self).__init__(*args, **kwargs)
        self.reparametrized = True

    def sample(self):
        """
        Reparameterized Normal cholesky sampler.
        """
        eps = Variable(torch.randn(self.mu.size()))
        if eps.dim() == 1:
            eps = eps.unsqueeze(1)
        z = self.mu + torch.mm(self.L, eps).squeeze()
        return z

    def log_pdf(self, x):
        """
        Normal cholesky log-likelihood
        """
        ll_1 = Variable(torch.Tensor([-0.5 * self.dim * np.log(2.0 * np.pi)]))
        ll_2 = -torch.sum(torch.log(torch.diag(self.L)))
        x_chol = Variable(
            torch.trtrs(
                (x - self.mu).unsqueeze(1).data,
                self.L.data,
                False)[0])
        ll_3 = -0.5 * torch.sum(torch.pow(x_chol, 2.0))

        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, batch_size=1):
        raise NotImplementedError()
