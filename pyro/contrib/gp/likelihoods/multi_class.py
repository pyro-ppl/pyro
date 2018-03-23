from __future__ import absolute_import, division, print_function

import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from .likelihood import Likelihood


def _softmax(x):
    return F.softmax(x, dim=-1)


class MultiClass(Likelihood):
    """
    Implementation of MultiClass likelihood, which is used for multi-class classification.
    """

    def __init__(self, response_function=None):
        super(MultiClass, self).__init__()
        self.response_function = (response_function if response_function is not None
                                  else _softmax)

    def forward(self, f, y):
        f_res = self.response_function(f)
        return pyro.sample("y", dist.Categorical(f_res.expand_as(y))
                           .reshape(extra_event_dims=y.dim()), obs=y)
