from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist

from .likelihood import Likelihood


class Poisson(Likelihood):
    """
    Implementation of Poisson likelihood, which is used for count data.

    Poisson likelihood uses the :class:`~pyro.distributions.Poisson`
    distribution, so the output of ``response_function`` should be positive.
    By default, we use :func:`torch.exp` as response function, corresponding
    to a log-Gaussian Cox process.

    :param callable response_function: A mapping to positive real numbers.
    """
    def __init__(self, response_function=None, name="Poisson"):
        super(Poisson, self).__init__(name)
        self.response_function = (torch.exp if response_function is None
                                  else response_function)

    def forward(self, f_loc, f_var, y=None):
        r"""
        Samples :math:`y` given :math:`f_{loc}`, :math:`f_{var}` according to

            .. math:: f & \sim \mathbb{Normal}(f_{loc}, f_{var}),\\
                y & \sim \mathbb{Poisson}(\exp(f)).

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

        y_dist = dist.Poisson(f_res)
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_res.dim()]).independent(y.dim())
        return pyro.sample(self.y_name, y_dist, obs=y)
