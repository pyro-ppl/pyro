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

def log_matrix_product(A, B):
    """
    Computes the matrix products of two matrices in log-space, returning the result in log-space.
    This is useful for calculating the vector chain rule for Jacobian terms.
    """
    return torch.logsumexp(A.unsqueeze(-1) + B.unsqueeze(-3), dim=-2)

# A first-version of Block-NAF with single hidden layers, and without residual connections/weight normalization

@copy_docs_from(TransformModule)
class BlockNAFFlow(TransformModule):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1
    autoregressive = True

    def __init__(self, input_dim):
        super(BlockNAFFlow, self).__init__(cache_size=1)

        # Initialize modules for each layer in flow
        self.input_dim = input_dim
        self.layers = nn.ModuleList([MaskedBlockLinear(input_dim, input_dim*2, input_dim), MaskedBlockLinear(input_dim*2, input_dim, input_dim)])
        self._cached_logDetJ = None

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        y = x
        for idx in range(len(self.layers)):
            y, dy_dx = self.layers[idx](y.unsqueeze(-1))
            #print('y', y.size(), 'dy_dx', dy_dx.size())

            if idx == 0:
                logDetJ = dy_dx
            else:
                logDetJ = log_matrix_product(dy_dx, logDetJ)

        self._cached_logDetJ = logDetJ.squeeze(-1).squeeze(-1)
        
        #print('logDetJ', self._cached_logDetJ.size())

        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """

        raise KeyError("BlockNAFFlow expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        return self._cached_logDetJ.sum(-1)


class MaskedBlockLinear(torch.nn.Module):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """

    def __init__(self, in_features, out_features, dim, bias=True):
        super(MaskedBlockLinear, self).__init__()
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

        # NOTE: Commented out weight normalization for now!
        # Sum is taken over columns, i.e. one norm per row
        #w_squared_norm = (w ** 2).sum(-1, keepdim=True)

        # Effect of multiplication and division is that each row is normalized and rescaled
        #w = self._diag_weight.exp() * w / w_squared_norm.sqrt()

        # NOTE: Not sure what's going on here... It's to do with the gradient of weight normalization
        #wpl = self._diag_weight + self._weight - 0.5 * torch.log(w_squared_norm)
        wpl = self._weight

        # TODO: Check I understand how wpl is working!
        return w, wpl[self.mask_d.byte()].view(self.dim, self.out_features // self.dim, self.in_features // self.dim)

    def forward(self, x):
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

        #g = wpl.transpose(-2, -1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)

        #print('w', w.size(), 'x', x.size(), 'bias', self.bias.size())
        return (torch.matmul(w, x) + self.bias.unsqueeze(-1)).squeeze(-1), wpl
        
        #torch.logsumexp(g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1) if grad is not None else g
