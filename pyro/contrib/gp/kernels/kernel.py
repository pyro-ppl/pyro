from __future__ import absolute_import, division, print_function

from pyro.contrib.gp import Parameterized


class Kernel(Parameterized):
    """
    Base class for kernels used in Gaussian Process.

    Every inherited class should implement the forward pass which
    take inputs X, X2 and return their covariance matrix.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        super(Kernel, self).__init__(name)
        if active_dims is None:
            active_dims = slice(input_dim)
        elif input_dim != len(active_dims):
            raise ValueError("Input size and the length of active dimensionals should be equal.")
        self.input_dim = input_dim
        self.active_dims = active_dims

    def forward(self, X, Z=None, diag=False):
        """
        Calculates covariance matrix of inputs on active dimensionals.

        :param torch.autograd.Variable X: A 2D tensor of size `N x input_dim`.
        :param torch.autograd.Variable Z: An optional 2D tensor of size `M x input_dim`.
        :param bool diag: A flag to decide if we want to return a full covariance matrix
            or just its diagonal part.
        :return: Covariance matrix of X and Z with size `N x M`.
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError

    def Kdiag(self, X):
        """
        Calculates the diagonal part of covariance matrix on active dimensionals.

        :param torch.autograd.Variable X: A 2D tensor of size `N x input_dim`.
        :return: Diagonal part of covariance matrix K(X, X).
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError

    def _slice_X(self, X):
        """
        Slices X according to `self.active_dims`. If X is 1 dimensional then returns
            a 2D tensor of size `N x 1`.

        :param torch.autograd.Variable X: A 1D or 2D tensor.
        :return: A 2D slice of X.
        :rtype: torch.autograd.Variable
        """
        if X.dim() == 2:
            return X[:, self.active_dims]
        elif X.dim() == 1:
            return X.unsqueeze(1)
        else:
            raise ValueError("Input X must be either 1 or 2 dimensional.")
