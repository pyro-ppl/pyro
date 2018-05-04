from __future__ import absolute_import, division, print_function

import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from .likelihood import Likelihood


class Binary(Likelihood):
    """
    Implementation of Binary likelihood, which is used for binary classification
    problems.

    Binary likelihood uses :class:`~pyro.distributions.Bernoulli` distribution,
    so the output of ``response_function`` should be in range :math:`(0,1)`. By
    default, we use `sigmoid` function.

    :param callable response_function: A mapping to correct domain for Binary
        likelihood.
    """
    def __init__(self, response_function=None, name="Binary"):
        super(Binary, self).__init__(name)
        self.response_function = (response_function if response_function is not None
                                  else F.sigmoid)

    def forward(self, f_loc, f_var, y=None):
        r"""
        Samples :math:`y` given :math:`f_{loc}`, :math:`f_{var}` according to

            .. math:: f & \sim \mathbb{Normal}(f_{loc}, f_{var}),\\
                y & \sim \mathbb{Bernoulli}(f).

        .. note:: The log likelihood is estimated using Monte Carlo with 1 sample of
            :math:`f`.

        :param torch.Tensor f_loc: Mean of latent function output.
        :param torch.Tensor f_var: Variance of latent function output.
        :param torch.Tensor y: Training output tensor.
        :returns: a tensor sampled from likelihood
        :rtype: torch.Tensor
        """
        # calculates Monte Carlo estimate for E_q(f) [logp(y | f)]
        f = dist.Normal(f_loc, f_var)()
        f_res = self.response_function(f)

        y_dist = dist.Bernoulli(f_res)
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_res.dim()]).independent(y.dim())
        return pyro.sample(self.y_name, y_dist, obs=y)
