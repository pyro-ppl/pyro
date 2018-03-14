from __future__ import absolute_import, division, print_function

import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from .likelihood import Likelihood


class Binary(Likelihood):
    """
    Implementation of Binary likelihood, which is used for binary classification.
    """

    def __init__(self, response_function=None):
        super(Binary, self).__init__()
        self.response_function = response_function if response_function is not None else F.sigmoid

    def forward(self, f, obs=None):
        event_dims = f.dim()
        f_response = self.response_function(f)
        return pyro.sample("y", dist.Bernoulli(f_response).reshape(extra_event_dims=event_dims),
                           obs=obs)
