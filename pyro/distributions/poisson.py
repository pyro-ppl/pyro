from __future__ import absolute_import, division, print_function

import numpy.random as npr
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_gamma


class Poisson(Distribution):
    """
    Poisson distribution over integers parameterized by scale `lambda`.

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
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        x = npr.poisson(lam=self.lam.data.cpu().numpy()).astype("float")
        return Variable(torch.Tensor(x).type_as(self.lam.data))

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        lam = self.lam.expand(self.shape(x))
        ll_1 = torch.sum(x * torch.log(lam), -1)
        ll_2 = -torch.sum(lam, -1)
        ll_3 = -torch.sum(log_gamma(x + 1.0), -1)
        batch_log_pdf = ll_1 + ll_2 + ll_3
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.lam

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return self.lam
