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

    def _check_Xnew_shape(self, Xnew, X):
        """
        Checks the correction of the shape of new data.
        """
        if Xnew.dim() != X.dim():
            raise ValueError("Train data and test data should have the same number of dimensions.")
        if Xnew.dim() == 2 and X.size(1) != Xnew.size(1):
            raise ValueError("Train data and test data should have the same feature sizes.")
