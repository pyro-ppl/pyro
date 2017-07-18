import numpy.random as npr
import torch
from torch.autograd import Variable
import pyro
from pyro.distributions.distribution import Distribution


def _log_gamma(xx):
    """
    quick and dirty log gamma copied from webppl
    """
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5
    ]
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = pyro.ones(x.size()) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


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
        ll_3 = -torch.sum(_log_gamma(x + 1.0))
        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, batch_size=1):
        if x.dim() == 1 and self.lam.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll_1 = torch.sum(x * torch.log(self.lam), 1)
        ll_2 = -torch.sum(self.lam, 1)
        ll_3 = -torch.sum(_log_gamma(x + 1.0), 1)
        return ll_1 + ll_2 + ll_3
