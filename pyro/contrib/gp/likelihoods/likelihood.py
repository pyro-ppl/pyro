from __future__ import absolute_import, division, print_function

from pyro.contrib.gp.util import Parameterized


class Likelihood(Parameterized):
    """
    Base class for likelihoods used in Gaussian Process.

    Every inherited class should implement a forward pass which
    takes an input :math:`f` and returns a sample :math:`y`.
    """
    def __init__(self):
        super(Likelihood, self).__init__(name="likelihood")

    def forward(self, f, y=None):
        """
        Samples :math:`y` given :math:`f`.

        :param torch.Tensor f: Latent function output tensor.
        :param torch.Tensor y: Training output tensor.
        :returns: A tensor sampled from likelihood.
        :rtype: torch.Tensor
        """
        raise NotImplementedError
