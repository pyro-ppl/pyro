from __future__ import absolute_import, division, print_function

from pyro.contrib.gp.util import Parameterized
from pyro.params import param_with_module_name


class Likelihood(Parameterized):
    """
    Base class for likelihoods used in Gaussian Process.

    Every inherited class should implement a forward pass which
    takes an input :math:`f` and returns a sample :math:`y`.
    """
    def __init__(self, name=None):
        super(Likelihood, self).__init__(name)
        self.y_name = (param_with_module_name(name, "y") if name is not None
                       else "y")

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
