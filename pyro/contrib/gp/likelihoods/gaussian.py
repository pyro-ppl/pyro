# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from pyro.contrib.gp.likelihoods.likelihood import Likelihood
from pyro.nn.module import PyroParam


class Gaussian(Likelihood):
    """
    Implementation of Gaussian likelihood, which is used for regression problems.

    Gaussian likelihood uses :class:`~pyro.distributions.Normal` distribution.

    :param torch.Tensor variance: A variance parameter, which plays the role of
        ``noise`` in regression problems.
    """
    def __init__(self, variance=None):
        super().__init__()

        variance = torch.tensor(1.) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def forward(self, f_loc, f_var, y=None):
        r"""
        Samples :math:`y` given :math:`f_{loc}`, :math:`f_{var}` according to

            .. math:: y \sim \mathbb{Normal}(f_{loc}, f_{var} + \epsilon),

        where :math:`\epsilon` is the ``variance`` parameter of this likelihood.

        :param torch.Tensor f_loc: Mean of latent function output.
        :param torch.Tensor f_var: Variance of latent function output.
        :param torch.Tensor y: Training output tensor.
        :returns: a tensor sampled from likelihood
        :rtype: torch.Tensor
        """
        y_var = f_var + self.variance

        y_dist = dist.Normal(f_loc, y_var.sqrt())
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)
