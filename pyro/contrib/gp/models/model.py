from __future__ import absolute_import, division, print_function

from pyro.contrib.gp import Parameterized


class Model(Parameterized):
    """
    Base class for models used in Gaussian Process.
    """

    def __init__(self):
        super(Model, self).__init__()

    def model(self):
        """
        A ``model`` stochastic method.
        """
        raise NotImplementedError

    def guide(self):
        """
        A ``guide`` stochastic method.
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Implements prediction step.
        """
        raise NotImplementedError
