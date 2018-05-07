from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    """
    A linear mapping with a given mask on the weights (arbitrary bias)

    :param in_features: the number of input features
    :type in_features: int
    :param out_features: the number of output features
    :type out_features: int
    :param mask: the mask to apply to the in_features x out_features weight matrix
    :type mask: torch.Tensor
    :param bias: whether or not `MaskedLinear` should include a bias term. defaults to `True`
    :type bias: bool
    """
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask.data)

    def forward(self, _input):
        """
        the forward method that does the masked linear computation and returns the result
        """
        masked_weight = self.weight * self.mask
        return F.linear(_input, masked_weight, self.bias)


class AutoRegressiveNN(nn.Module):
    """
    A simple implementation of a MADE-like auto-regressive neural network.

    Reference:
    MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param hidden_dim: the dimensionality of the hidden units
    :type hidden_dim: int
    :param output_dim_multiplier: the dimensionality of the output is given by input_dim x output_dim_multiplier.
        specifically the shape of the output for a single vector input is [output_dim_multiplier, input_dim].
        for any i, j in range(0, output_dim_multiplier) the subset of outputs [i, :] has identical
        autoregressive structure to [j, :]. defaults to `1`
    :type output_dim_multiplier: int
    :param mask_encoding: a torch Tensor that controls the autoregressive structure (see reference). by default
        this is chosen at random.
    :type mask_encoding: torch.LongTensor
    :param permutation: an optional permutation that is applied to the inputs and controls the order of the
        autoregressive factorization. in particular for the identity permutation the autoregressive structure
        is such that the Jacobian is upper triangular. by default this is chosen at random.
    :type permutation: torch.LongTensor
    """

    def __init__(self, input_dim, hidden_dim, output_dim_multiplier=1,
                 mask_encoding=None, permutation=None):
        super(AutoRegressiveNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim_multiplier = output_dim_multiplier

        if mask_encoding is None:
            # the dependency structure is chosen at random
            self.mask_encoding = 1 + torch.multinomial(torch.ones(input_dim - 1) / (input_dim - 1),
                                                       num_samples=hidden_dim, replacement=True)
        else:
            # the dependency structure is given by the user
            self.mask_encoding = mask_encoding

        if permutation is None:
            # a permutation is chosen at random
            self.permutation = torch.randperm(input_dim, device=torch.device('cpu'))
        else:
            # the permutation is chosen by the user
            self.permutation = permutation

        # these masks control the autoregressive structure
        self.mask1 = torch.zeros(hidden_dim, input_dim)
        self.mask2 = torch.zeros(input_dim * self.output_dim_multiplier, hidden_dim)

        for k in range(hidden_dim):
            # fill in mask1
            m_k = self.mask_encoding[k].item()
            slice_k = torch.cat([torch.ones(m_k), torch.zeros(input_dim - m_k)])
            for j in range(input_dim):
                self.mask1[k, self.permutation[j]] = slice_k[j]
            # fill in mask2
            slice_k = torch.cat([torch.zeros(m_k), torch.ones(input_dim - m_k)])
            for r in range(self.output_dim_multiplier):
                for j in range(input_dim):
                    self.mask2[r * input_dim + self.permutation[j], k] = slice_k[j]

        self.lin1 = MaskedLinear(input_dim, hidden_dim, self.mask1)
        self.lin2 = MaskedLinear(hidden_dim, input_dim * output_dim_multiplier, self.mask2)
        self.relu = nn.ReLU()

    def get_mask_encoding(self):
        """
        Get the mask encoding associated with the neural network: basically the quantity m(k) in the MADE paper.
        """
        return self.mask_encoding

    def get_permutation(self):
        """
        Get the permutation applied to the inputs (by default this is chosen at random)
        """
        return self.permutation

    def forward(self, z):
        """
        the forward method
        """
        h = self.relu(self.lin1(z))
        out = self.lin2(h)
        return out
