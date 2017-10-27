import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Uniform(Distribution):
    """
    :param a: lower bound *(real)*
    :param b: upper bound (>a) *(real)*

    Continuous uniform distribution over ``[a, b]``
    """
    reparameterized = False  # XXX Why is this marked non-differentiable?

    def __init__(self, a, b, batch_size=None, *args, **kwargs):
        """
        Params:
          `a` - low bound
          `b` -  high bound
        """
        if a.size() != b.size():
            raise ValueError("Expected a.size() == b.size(), but got {} vs {}"
                             .format(a.size(), b.size()))
        self.a = a
        self.b = b
        if a.dim() == 1 and batch_size is not None:
            self.a = a.expand(batch_size, a.size(0))
            self.b = b.expand(batch_size, b.size(0))
        super(Uniform, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        event_dim = 1
        a = self.a
        if x is not None and x.size() != a.size():
            a = self.a.expand_as(x)
        return a.size()[:-event_dim]

    def event_shape(self):
        event_dim = 1
        return self.a.size()[-event_dim:]

    def shape(self, x=None):
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        """
        Reparameterized Uniform sampler.
        """
        eps = Variable(torch.rand(self.a.size()).type_as(self.a.data))
        return self.a + torch.mul(eps, self.b - self.a)

    def batch_log_pdf(self, x):
        a = self.a.expand(self.shape(x))
        b = self.b.expand(self.shape(x))
        lb = x.ge(a).type_as(a)
        ub = x.le(b).type_as(b)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return torch.sum(torch.log(lb.mul(ub)) - torch.log(b - a), -1).contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        return 0.5 * (self.a + self.b)

    def analytic_var(self):
        return torch.pow(self.b - self.a, 2) / 12
