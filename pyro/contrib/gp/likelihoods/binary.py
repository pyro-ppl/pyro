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
    def __init__(self, response_function=None, name="Binary"):
        super(Binary, self).__init__(name)
        self.response_function = (response_function if response_function is not None
                                  else F.sigmoid)

    def forward(self, f_loc, f_var, y):
        # calculates Monte Carlo estimate for E_q(f) [logp(y | f)]
        f = dist.Normal(f_loc, f_var)()
        f_res = self.response_function(f)
        print(f_res.shape, y.shape)
        return pyro.sample(self.y_name,
                           dist.Bernoulli(f_res)
                               .reshape(sample_shape=y.shape[:-f_res.dim()],
                                        extra_event_dims=y.dim()),
                           obs=y)
