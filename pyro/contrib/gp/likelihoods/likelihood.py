# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pyro.contrib.gp.parameterized import Parameterized


class Likelihood(Parameterized):
    """
    Base class for likelihoods used in Gaussian Process.

    Every inherited class should implement a forward pass which
    takes an input :math:`f` and returns a sample :math:`y`.
    """
    def __init__(self):
        super().__init__()

    def forward(self, f_loc, f_var, y=None):
        """
        Samples :math:`y` given :math:`f_{loc}`, :math:`f_{var}`.

        :param torch.Tensor f_loc: Mean of latent function output.
        :param torch.Tensor f_var: Variance of latent function output.
        :param torch.Tensor y: Training output tensor.
        :returns: a tensor sampled from likelihood
        :rtype: torch.Tensor
        """
        raise NotImplementedError
