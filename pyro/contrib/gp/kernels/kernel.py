from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Kernel(nn.Module):
    """
    Base class for kernels used in Gaussian Process.

    Every inherited class should implement the forward pass which
        take inputs X, X2 and return their covariance matrix.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        super(Kernel, self).__init__()
        if active_dims is None:
            active_dims = slice(input_dim)
        elif input_dim != len(active_dims):
            raise ValueError("Input size and the length of active dimensionals should be equal.")
        self.input_dim = input_dim
        self.active_dims = active_dims
        self.name = name

    def forward(self, X, Z=None):
        """
        Calculate covariance matrix of inputs on active dimensionals.

        :param torch.autograd.Variable X: A 2D tensor of size `N x input_dim`.
        :param torch.autograd.Variable Z: A 2D tensor of size `N x input_dim`.
        :return: Covariance matrix of X and Z with size `N x N`.
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError

    def _slice_X(self, X):
        """
        Slice X according to `self.active_dims`. If X is 1 dimensional then returns
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
