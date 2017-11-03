from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Delta(Distribution):
    """
    Degenerate discrete distribution (a single point).

    Discrete distribution that assigns probability one to the single element in
    its support. Delta distribution parameterized by a random choice should not
    be used with MCMC based inference, as doing so produces incorrect results.

    :param torch.autograd.Variable v: The single support element.
    """
    enumerable = True

    def __init__(self, v, batch_size=None, *args, **kwargs):
        self.v = v
        if not isinstance(self.v, Variable):
            self.v = Variable(self.v)
        if v.dim() == 1 and batch_size is not None:
            self.v = v.expand(v, v.size(0))
        super(Delta, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        v = self.v
        if x is not None:
            if x.size()[-event_dim] != v.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.v.size()[-1], but got {} vs {}".format(
                                     x.size(-1), v.size(-1)))
            try:
                v = self.v.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `v` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(v.size(), x.size(), str(e)))
        return v.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        event_dim = 1
        return self.v.size()[-event_dim:]

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        return self.v

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        v = self.v
        v = v.expand(self.shape(x))
        batch_shape = self.batch_shape(x) + (1,)
        return torch.sum(torch.eq(x, v).float().log(), -1).contiguous().view(batch_shape)

    def enumerate_support(self, v=None):
        """
        Returns the delta distribution's support, as a tensor along the first dimension.

        :param v: torch variable where each element of the tensor represents the point at
            which the delta distribution is concentrated.
        :return: torch variable enumerating the support of the delta distribution.
        :rtype: torch.autograd.Variable.
        """
        return Variable(self.v.data)
