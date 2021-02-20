# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from pyro.contrib.gp.likelihoods.likelihood import Likelihood


def _softmax(x):
    return F.softmax(x, dim=-1)


class MultiClass(Likelihood):
    """
    Implementation of MultiClass likelihood, which is used for multi-class
    classification problems.

    MultiClass likelihood uses :class:`~pyro.distributions.Categorical`
    distribution, so ``response_function`` should normalize its input's rightmost axis.
    By default, we use `softmax` function.

    :param int num_classes: Number of classes for prediction.
    :param callable response_function: A mapping to correct domain for MultiClass
        likelihood.
    """
    def __init__(self, num_classes, response_function=None):
        super().__init__()
        self.num_classes = num_classes
        self.response_function = _softmax if response_function is None else response_function

    def forward(self, f_loc, f_var, y=None):
        r"""
        Samples :math:`y` given :math:`f_{loc}`, :math:`f_{var}` according to

            .. math:: f & \sim \mathbb{Normal}(f_{loc}, f_{var}),\\
                y & \sim \mathbb{Categorical}(f).

        .. note:: The log likelihood is estimated using Monte Carlo with 1 sample of
            :math:`f`.

        :param torch.Tensor f_loc: Mean of latent function output.
        :param torch.Tensor f_var: Variance of latent function output.
        :param torch.Tensor y: Training output tensor.
        :returns: a tensor sampled from likelihood
        :rtype: torch.Tensor
        """
        # calculates Monte Carlo estimate for E_q(f) [logp(y | f)]
        f = dist.Normal(f_loc, f_var.sqrt())()
        if f.dim() < 2:
            raise ValueError("Latent function output should have at least 2 "
                             "dimensions: one for number of classes and one for "
                             "number of data.")

        # swap class dimension and data dimension
        f_swap = f.transpose(-2, -1)  # -> num_data x num_classes
        if f_swap.size(-1) != self.num_classes:
            raise ValueError("Number of Gaussian processes should be equal to the "
                             "number of classes. Expected {} but got {}."
                             .format(self.num_classes, f_swap.size(-1)))
        if self.response_function is _softmax:
            y_dist = dist.Categorical(logits=f_swap)
        else:
            f_res = self.response_function(f_swap)
            y_dist = dist.Categorical(f_res)
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f.dim() + 1]).to_event(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)
