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

    def forward(self, f, obs=None):
        """
        Samples :math:`y` (``obs``) given :math:`f`.

        :param torch.Tensor f: A 1D tensor of size :math:`N`.
        :param torch.Tensor obs: A 1D tensor of size :math:`N`.
        :return: A 1D tensor of size :math:`N`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError
