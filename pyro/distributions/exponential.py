import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Exponential(Distribution):
    """
    Exponential parameterized by scale `lambda`.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `lam` parameter is positive.

    :param torch.autograd.Variable lam: Scale parameter (a.k.a. `lambda`).
        Should be positive.
    """
    reparameterized = True

    def __init__(self, lam, batch_size=None, *args, **kwargs):
        self.lam = lam
        if lam.dim() == 1 and batch_size is not None:
            self.lam = lam.expand(batch_size, lam.size(0))
        super(Exponential, self).__init__(*args, **kwargs)

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
        """
        Reparameterized sampler.
        """
        eps = Variable(torch.rand(self.lam.size()).type_as(self.lam.data))
        x = -torch.log(eps) / self.lam
        return x

    def batch_log_pdf(self, x):
        lam = self.lam.expand_as(x)
        ll = - lam * x + torch.log(lam)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return torch.sum(ll, -1).contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        return torch.pow(self.lam, -1.0)

    def analytic_var(self):
        return torch.pow(self.lam, -2.0)
