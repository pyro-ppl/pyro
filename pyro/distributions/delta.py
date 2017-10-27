import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Delta(Distribution):
    """
    :param v: support element *(any)*

    Discrete distribution that assigns probability one to the single element in
    its support. Delta distribution parameterized by a random choice should not
    be used with MCMC based inference, as doing so produces incorrect results.
    """
    enumerable = True

    def __init__(self, v, batch_size=None, *args, **kwargs):
        """
        Params:
          `v` - value
        """
        self.v = v
        if not isinstance(self.v, Variable):
            self.v = Variable(self.v)
        if v.dim() == 1 and batch_size is not None:
            self.v = v.expand(v, v.size(0))
        super(Delta, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        event_dim = 1
        v = self.v
        if x is not None and x.size() != v.size():
            v = self.v.expand_as(x)
        return v.size()[:-event_dim]

    def event_shape(self):
        event_dim = 1
        return self.v.size()[-event_dim:]

    def shape(self, x=None):
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        return self.v

    def batch_log_pdf(self, x):
        v = self.v
        if x.size() != v.size():
            v = v.expand_as(x)
        return torch.sum(torch.eq(x, v).float().log(), -1)

    def support(self, v=None):
        """
        Returns the delta distribution's support, as a tensor along the first dimension.

        :param v: torch variable where each element of the tensor represents the point at
            which the delta distribution is concentrated.
        :return: torch variable enumerating the support of the delta distribution.
        :rtype: torch.autograd.Variable.
        """
        return Variable(self.v.data)
