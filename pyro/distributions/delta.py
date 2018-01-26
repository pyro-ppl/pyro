from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from, broadcast_shape


@copy_docs_from(Distribution)
class Delta(Distribution):
    """
    Degenerate discrete distribution (a single point).

    Discrete distribution that assigns probability one to the single element in
    its support. Delta distribution parameterized by a random choice should not
    be used with MCMC based inference, as doing so produces incorrect results.

    :param torch.autograd.Variable v: The single support element.
    """
    enumerable = True

    def __init__(self, v, *args, **kwargs):
        self.v = v
        if not isinstance(self.v, Variable):
            self.v = Variable(self.v)
        super(Delta, self).__init__(*args, **kwargs)

    def batch_shape(self):
        return self.v.size()

    def event_shape(self):
        return torch.Size()

    def sample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.v.size()
        return self.v.expand(shape)

    def log_prob(self, x):
        v = self.v
        v = v.expand(broadcast_shape(self.shape(), x.size()))
        return torch.eq(x, v).float().log()

    def enumerate_support(self, v=None):
        """
        Returns the delta distribution's support, as a tensor along the first dimension.

        :param v: torch variable where each element of the tensor represents the point at
            which the delta distribution is concentrated.
        :return: torch variable enumerating the support of the delta distribution.
        :rtype: torch.autograd.Variable.
        """
        return Variable(self.v.data.unsqueeze(0))
