import torch
import itertools as it
import functools as ft
from torch.autograd import Variable
from torch import Tensor as T
from pyro.distributions.distribution import Distribution


class Bernoulli(Distribution):
    """
    Multi-variate bernoulli
    Params:
      ps = tensor of probabilities
    """

    def __init__(self, ps, batch_size=1, *args, **kwargs):
        """
        Constructor.
        """
        if ps.dim() == 1:
            self.ps = ps.expand(batch_size, ps.size(0))
        else:
            self.ps = ps
        super(Bernoulli, self).__init__(*args, **kwargs)

    def sample(self, batch_size=1):
        """
        Reparameterized Bernoulli sampler.
        """
        if batch_size != 1 and batch_size != self.bs:
            raise ValueError("Batch sizes do not match")
        return torch.bernoulli(self.ps)

    def log_pdf(self, x, batch_size=1):
        """
        log-likelihood
        """
        x_1 = x - 1
        ps_1 = self.ps - 1
        xmul = torch.mul(x, self.ps)
        xmul_1 = torch.mul(x_1, ps_1)
        logsum = torch.log(torch.add(xmul, xmul_1))
        return torch.sum(logsum)

    def batch_log_pdf(self, x, batch_size=1):
        if x.dim() == 1 and self.ps.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        x_1 = x - 1
        ps_1 = self.ps - 1
        xmul = torch.mul(x, self.ps)
        xmul_1 = torch.mul(x_1, ps_1)
        logsum = torch.log(torch.add(xmul, xmul_1))
        return torch.sum(logsum, 1)

    def support(self):
        if self.ps.dim() == 1:
            return iter([Variable(torch.ones(1)), Variable(torch.zeros(1))])
        size = ft.reduce(lambda x, y: x * y, self.ps.size())
        return (Variable(T(list(x)).view_as(self.ps)) for x in it.product(T([0, 1]), repeat=size))
