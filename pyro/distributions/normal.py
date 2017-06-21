import numpy as np
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution


class Normal(Distribution):
    """
    Multi-variate normal with arbitrary covariance sigma
    parameterized by its mean mu and covariance matrix sigma
    """

    def __init__(self, mu, sigma, batch_size=1, *args, **kwargs):
        """
        Constructor.
        """
        self.dim = mu.size(0)
        self.l_chol = Variable(torch.from_numpy(self.do_cholesky(sigma)))
        if batch_size == 1:
            self.mu = torch.unsqueeze(mu, 0)
            self.sigma = torch.unsqueeze(sigma, 0)
        else:
            self.mu = mu.expand(batch_size, mu.size(0))
            self.sigma = sigma.expand(batch_size, sigma.size(0))
        super(Normal, self).__init__(*args, **kwargs)
        self.reparametrized = True

    def do_cholesky(self, sigma, batch_size=1):
        # do a cholesky decomposition
        if batch_size != 1 and batch_size != self.bs:
            raise ValueError("Batch sizes do not match")
        L = np.linalg.cholesky(sigma.data.cpu().numpy())
        return L

    def sample(self):
        """
        Reparameterized Normal sampler.
        """
        eps = Variable(torch.randn(self.mu.size()),
                       requires_grad=False).type_as(self.mu)
        z = self.mu + torch.mm(self.l_chol, eps.unsqueeze(1)).squeeze()
        return z

    def log_pdf(self, x):
        """
        Normal log-likelihood
        """
#         x_chol dims needs to be fixed since x-.self.mu dims is not nx1 right now
#         if x.dim() == 1
#             x = x.expand(batch_size, x.size(0))
        ll_1 = Variable(torch.Tensor([-0.5 * self.dim * np.log(2.0 * np.pi)]))
        ll_2 = -torch.sum(torch.log(torch.diag(self.l_chol)))
        x_chol = Variable(
            torch.trtrs(
                (x - self.mu).data,
                self.l_chol.data,
                False)[0])
        ll_3 = -0.5 * torch.sum(torch.pow(x_chol, 2.0))

        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, batch_size=1):
        raise NotImplementedError()
