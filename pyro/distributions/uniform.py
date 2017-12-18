from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Uniform(Distribution):
    """
    Uniform distribution over the continuous interval `[a, b]`.

    :param torch.autograd.Variable a: lower bound (real).
    :param torch.autograd.Variable b: upper bound (real).
        Should be greater than `a`.
    """
    reparameterized = False  # XXX Why is this marked non-differentiable?

    def __init__(self, a, b, batch_size=None, *args, **kwargs):
        if a.size() != b.size():
            raise ValueError("Expected a.size() == b.size(), but got {} vs {}".format(a.size(), b.size()))
        self.a = a
        self.b = b
        if a.dim() == 1 and batch_size is not None:
            self.a = a.expand(batch_size, a.size(0))
            self.b = b.expand(batch_size, b.size(0))
        super(Uniform, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        a = self.a
        if x is not None:
            if x.size()[-event_dim] != a.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.a.size()[-1], but got {} vs {}".format(
                                     x.size(-1), a.size(-1)))
            try:
                a = self.a.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `a` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(a.size(), x.size(), str(e)))
        return a.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        event_dim = 1
        return self.a.size()[-event_dim:]

    def shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.shape`
        """
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        eps = Variable(torch.rand(self.a.size()).type_as(self.a.data))
        return self.a + torch.mul(eps, self.b - self.a)

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        a = self.a.expand(self.shape(x))
        b = self.b.expand(self.shape(x))
        lb = x.ge(a).type_as(a)
        ub = x.le(b).type_as(b)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return torch.sum(torch.log(lb.mul(ub)) - torch.log(b - a), -1).contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return 0.5 * (self.a + self.b)

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return torch.pow(self.b - self.a, 2) / 12
