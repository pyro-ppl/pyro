from __future__ import absolute_import, division, print_function

import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from .likelihood import Likelihood


class Binary(Likelihood):
    """
    Implementation of Binary likelihood, which is used for binary classification.

    :param callable response_function: A mapping to correct domain for Binary likelihood.
        By default, we use `sigmoid` function.
    """
    def __init__(self, response_function=None):
        super(Binary, self).__init__()
        self.response_function = (response_function if response_function is not None
                                  else F.sigmoid)

    def forward(self, f, y=None):
        f_res = self.response_function(f)
        if y is None:
            return pyro.sample("y", dist.Bernoulli(f_res))
        else:
            return pyro.sample("y", dist.Bernoulli(f_res.expand_as(y))
                               .reshape(extra_event_dims=y.dim()), obs=y)
