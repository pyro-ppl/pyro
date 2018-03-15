from __future__ import absolute_import, division, print_function

from pyro.contrib.gp.util import Parameterized
from pyro.infer import SVI
from pyro.optim import Adam, PyroOptim


class Model(Parameterized):
    """
    Base class for models used in Gaussian Process.
    """

    def __init__(self):
        super(Model, self).__init__()

    def set_data(self, X, y):
        """
        Sets data for Gaussian Process models.

        :param torch.Tensor X: A 1D or 2D tensor of input data for training.
        :param torch.Tensor y: A 1D or 2D tensor of output data for training.
        """
        if X.dim() > 2 or y.dim() > 2:
            raise ValueError("Input tensor and output tensor should be of 1 or 2 dimensionals.")
        if X.size(0) != y.size(0):
            raise ValueError("Expect the number of data inputs equal to the number of data "
                             "outputs, but got {} and {}.".format(X.size(0), y.size(0)))
        self.X = X
        self.y = y

    def model(self):
        """
        A "model" stochastic method.
        """
        raise NotImplementedError

    def guide(self):
        """
        A "guide" stochastic method.
        """
        raise NotImplementedError

    def optimize(self, optimizer=Adam({}), num_steps=1000):
        """
        A convenient method to optimize parameters for the Gaussian Process model
        using SVI.

        :param pyro.optim.PyroOptim optimizer: Optimizer.
        :param int num_steps: Number of steps to run SVI.
        :returns: losses of the training procedure
        :rtype: list
        """
        if not isinstance(optimizer, PyroOptim):
            raise ValueError("Optimizer should be an instance of pyro.optim.PyroOptim class.")
        svi = SVI(self.model, self.guide, optimizer, loss="ELBO")
        losses = []
        for i in range(num_steps):
            losses.append(svi.step())
        return losses

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
