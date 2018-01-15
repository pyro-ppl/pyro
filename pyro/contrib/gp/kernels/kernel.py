from __future__ import absolute_import, division, print_function

import torch.nn as nn


class Kernel(nn.Module):
    """
    Base class for kernels used in Gaussian Process.

    Every inherited class should implement the forward pass which
        take inputs X, X2 and return their covariance matrix.
    """

    def __init__(self, active_dims=None, name=None):
        super(Kernel, self).__init__()
        self.active_dims = active_dims
        self.name = name

    def K(self, X, Z=None):
        """
        Calculate covariance matrix of inputs on active dimensionals.

        :param torch.autograd.Variable X: A 2D tensor of size `N x input_dim`.
        :param torch.autograd.Variable X2: A 2D tensor of size `N x input_dim`.
        :return: Covariance matrix of X and X2 with size `N x N`.
        :rtype: torch.autograd.Variable
        """
        if Z is None:
            Z = X
        X = self._slice_X(X)
        Z = self._slice_X(Z)
        K = self(X, Z)
        return K

    def _slice_X(self, X):
        """
        :param torch.autograd.Variable X: A 2D tensor.
        :return: Slice X according to `self.active_dims`.
        :rtype: torch.autograd.Variable
        """
        if X.dim() == 2:
            active_dims = self.active_dims
            if active_dims is None:
                active_dims = slice(X.size(1))
            return X[:, active_dims]
        elif X.dim() == 1:
            return X.unsqueeze(1)
        else:
            raise ValueError("Input X must be either 1 or 2 dimensional.")
