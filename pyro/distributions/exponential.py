from __future__ import absolute_import, division, print_function

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
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        lam = self.lam
        if x is not None:
            if x.size()[-event_dim] != lam.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.lam.size()[-1], but got {} vs {}".format(
                                     x.size(-1), lam.size(-1)))
            try:
                lam = self.lam.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `lam` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(lam.size(), x.size(), str(e)))
        return lam.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        event_dim = 1
        return self.lam.size()[-event_dim:]

    def sample(self):
        """
        Reparameterized sampler.

        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        eps = Variable(torch.rand(self.lam.size()).type_as(self.lam.data))
        x = -torch.log(eps) / self.lam
        return x

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        lam = self.lam.expand(self.shape(x))
        ll = -lam * x + torch.log(lam)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return torch.sum(ll, -1).contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return torch.pow(self.lam, -1.0)

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return torch.pow(self.lam, -2.0)
