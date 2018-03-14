from __future__ import absolute_import, division, print_function

import pyro
import pyro.distributions as dist

from .likelihood import Likelihood


class Binary(Likelihood):
    """
    Implementation of Binary likelihood, which is used for binary classification.
    """

    def __init__(self, response_function=None):
        super(Binary, self).__init__()
        self.response_function = response_function

    def forward(self, f, obs=None):
        event_dims = f.dim()
        return pyro.sample("y", dist.Binomial(self.response_function(f)).reshape(extra_event_dims=event_dims),
                           obs=obs)
