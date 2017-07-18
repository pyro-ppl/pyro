import numpy.random as npr
import torch
from torch.autograd import Variable
import pyro
from pyro.distributions.distribution import Distribution
from pyro.util import log_gamma


class Poisson(Distribution):
    """
    Multi-variate poisson parameterized by its mean lam
    """

    def __init__(self, lam, batch_size=1, *args, **kwargs):
        """
        Constructor.
        """
        if lam.dim() == 1 and batch_size > 1:
            self.lam = lam.unsqueeze(0).expand(batch_size, lam.size(0))
        else:
            self.lam = lam
        super(Poisson, self).__init__(*args, **kwargs)

    def sample(self, batch_size=1):
        """
        Poisson sampler.
        """
        x = npr.poisson(lam=self.lam.data.numpy()).astype("float")
        return Variable(torch.Tensor(x))

    def log_pdf(self, x, batch_size=1):
        """
        Poisson log-likelihood
        warning: need pytorch implementation of log gamma in order to be ADable
        """
        ll_1 = torch.sum(x * torch.log(self.lam))
        ll_2 = -torch.sum(self.lam)
        ll_3 = -torch.sum(log_gamma(x + 1.0))
        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, batch_size=1):
        if x.dim() == 1 and self.lam.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll_1 = torch.sum(x * torch.log(self.lam), 1)
        ll_2 = -torch.sum(self.lam, 1)
        ll_3 = -torch.sum(log_gamma(x + 1.0), 1)
        return ll_1 + ll_2 + ll_3
