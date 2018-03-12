from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class Delta(TorchDistribution):
    """
    Degenerate discrete distribution (a single point).

    Discrete distribution that assigns probability one to the single element in
    its support. Delta distribution parameterized by a random choice should not
    be used with MCMC based inference, as doing so produces incorrect results.

    :param torch.Tensor v: The single support element.
    """
    has_rsample = True
    has_enumerate_support = True
    event_shape = torch.Size()

    def __init__(self, v, *args, **kwargs):
        self.v = v
        super(Delta, self).__init__(*args, **kwargs)

    @property
    def batch_shape(self):
        return self.v.size()

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.v.size()
        return self.v.expand(shape)

    def log_prob(self, x):
        v = self.v
        v = v.expand(broadcast_shape(self.shape(), x.size()))
        return torch.eq(x, v).float().log()

    def enumerate_support(self, v=None):
        """
        Returns the delta distribution's support, as a tensor along the first dimension.

        :param v: torch tensor where each element of the tensor represents the point at
            which the delta distribution is concentrated.
        :return: torch tensor enumerating the support of the delta distribution.
        :rtype: torch.Tensor.
        """
        return torch.tensor(self.v.data.unsqueeze(0))

    @property
    def mean(self):
        return self.v

    @property
    def variance(self):
        return torch.zeros_like(self.v)
