from __future__ import absolute_import, division, print_function

import math
import sys

import torch
import torch.nn as nn
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import constraints
import torch.nn.functional as F

from pyro.distributions.util import copy_docs_from

eps = 1e-6


class MaskedWeight(torch.nn.Module):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """

    def __init__(self, in_features, out_features, dim, bias=True):
        super(MaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim

        weight = torch.zeros(out_features, in_features)

        # Fill in non-zero entries of block weight matrix, going from top
        # to bottom.
        for i in range(dim):
            weight[i * out_features // dim:(i + 1) * out_features // dim,
                   0:(i + 1) * in_features // dim] = torch.nn.init.xavier_uniform_(
                torch.Tensor(out_features // dim, (i + 1) * in_features // dim))

        self._weight = torch.nn.Parameter(weight)
        self._diag_weight = torch.nn.Parameter(torch.nn.init.uniform_(torch.Tensor(out_features, 1)).log())

        self.bias = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.Tensor(out_features),
                                   -1 / math.sqrt(out_features),
                                   1 / math.sqrt(out_features))) if bias else 0

        # Diagonal block mask
        mask_d = torch.eye(dim).unsqueeze(-1).repeat(1, out_features // dim, in_features // dim).view(out_features, in_features)
        self.register_buffer('mask_d', mask_d)

        # Off-diagonal block mask for lower triangular weight matrix
        mask_o = torch.tril(torch.ones(dim, dim), diagonal=-1).unsqueeze(-1).repeat(1, out_features // dim, in_features // dim).view(out_features, in_features)
        self.register_buffer('mask_o', mask_o)

    def get_weights(self):
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """

        # Form block weight matrix, making sure it's positive on diagonal!
        w = torch.exp(self._weight) * self.mask_d + self._weight * self.mask_o

        # Sum is taken over columns, i.e. one norm per row
        w_squared_norm = (w ** 2).sum(-1, keepdim=True)

        # Effect of multiplication and division is that each row is normalized and rescaled
        w = self._diag_weight.exp() * w / w_squared_norm.sqrt()

        # NOTE: Not sure what's going on here...
        wpl = self._diag_weight + self._weight - 0.5 * torch.log(w_squared_norm)

        return w.t(), wpl.t()[self.mask_d.byte().t()].view(
            self.dim, self.in_features // self.dim, self.out_features // self.dim)

    def forward(self, inputs, grad: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal block of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """

        w, wpl = self.get_weights()

        g = wpl.transpose(-2, -1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)

        return inputs.matmul(w) + self.bias, torch.logsumexp(
            g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1) if grad is not None else g

    def __repr__(self):
        return 'MaskedWeight(in_features={}, out_features={}, dim={}, bias={})'.format(
            self.in_features, self.out_features, self.dim, not isinstance(self.bias, int))
