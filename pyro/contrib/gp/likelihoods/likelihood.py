from __future__ import absolute_import, division, print_function

from pyro.contrib.gp import Parameterized


class Likelihood(Parameterized):
    """
    Base class for likelihoods used in Gaussian Process.

    Every inherited class should implement a forward pass which
        takes an input `f` and returns a sample `y`.
    """

    def __init__(self):
        super(Likelihood, self).__init__(name="likelihood")

    def forward(self, f, obs=None):
        """
        Samples `y` given `f`.

        :param torch.autograd.Variable f: A 1D tensor of size `N`.
        :return: A 1D tensor of size `N`.
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError
