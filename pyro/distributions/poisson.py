import numpy.random as npr
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_gamma


class Poisson(Distribution):
    """
    Poisson distribution over integers parameterizeds by scale `lambda`.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `lam` parameter is positive.

    :param torch.autograd.Variable lam: Mean parameter (a.k.a. `lambda`).
        Should be positive.
    """

    def __init__(self, lam, batch_size=None, *args, **kwargs):
        self.lam = lam
        if lam.dim() == 1 and batch_size is not None:
            self.lam = lam.expand(batch_size, lam.size(0))
        super(Poisson, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        event_dim = 1
        lam = self.lam
        if x is not None and x.size() != lam.size():
            lam = self.lam.expand_as(x)
        return lam.size()[:-event_dim]

    def event_shape(self):
        event_dim = 1
        return self.lam.size()[-event_dim:]

    def shape(self, x=None):
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        x = npr.poisson(lam=self.lam.data.cpu().numpy()).astype("float")
        return Variable(torch.Tensor(x).type_as(self.lam.data))

    def batch_log_pdf(self, x):
        lam = self.lam.expand(self.shape(x))
        ll_1 = torch.sum(x * torch.log(lam), -1)
        ll_2 = -torch.sum(lam, -1)
        ll_3 = -torch.sum(log_gamma(x + 1.0), -1)
        batch_log_pdf = ll_1 + ll_2 + ll_3
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        return self.lam

    def analytic_var(self):
        return self.lam
