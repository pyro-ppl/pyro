import numpy as np
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution


class LogNormal(Distribution):
    """
    uni-variate normal
    parameterized by its mean mu and std sigma
    """

    def __init__(self, mu, sigma, batch_size=1, *args, **kwargs):
        """
        Constructor.
        """
        if mu.dim() != sigma.dim():
            raise ValueError("Mu and sigma need to have the same dimensions.")
        elif mu.dim() == 1:
            self.mu = mu.unsqueeze(0).expand(batch_size, 0)
            self.sigma = sigma.unsqueeze(0).expand(batch_size, 0)
        else:
            self.mu = mu
            self.sigma = sigma
        super(LogNormal, self).__init__(*args, **kwargs)
        self.reparametrized = True

    def sample(self):
        """
        Reparameterized log-normal sampler.
        """
        eps = Variable(torch.randn(1),
                       requires_grad=False).type_as(self.mu)
        z = self.mu + self.sigma * eps
        return torch.exp(z)

    def log_pdf(self, x):
        """
        log-normal log-likelihood
        """
        ll_1 = Variable(torch.Tensor([-0.5 * np.log(2.0 * np.pi)]))
        ll_2 = -torch.log(self.sigma * x)
        ll_3 = -0.5 * torch.pow((torch.log(x) - self.mu) / self.sigma, 2.0)
        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, batch_size=1):
        """
        log-normal log-likelihood
        """
        if x.dim() == 1 and self.mu.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll_1 = Variable(torch.Tensor([-0.5 * np.log(2.0 * np.pi)]).expand_as(x))
        ll_2 = -torch.log(self.sigma * x)
        ll_3 = -0.5 * torch.pow((torch.log(x) - self.mu) / self.sigma, 2.0)
        return ll_1 + ll_2 + ll_3

    def support(self):
        raise NotImplementedError("Support not supported for continuous distributions")
