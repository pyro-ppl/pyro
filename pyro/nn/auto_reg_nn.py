import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from pyro.util import ng_ones


class MaskedLinear(nn.Linear):
    """
    a linear mapping with a given mask on the weights (arbitrary bias)
    """

    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask = mask

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)


class AutoRegressiveNN(nn.Module):
    """
    A simple implementation of a MADE-like auto-regressive neural network
    The vector mask_encoding of dimensionality input_dim encodes the binary masks
    that encode the allowed dependencies
    reference: https://arxiv.org/abs/1502.03509
    """

    def __init__(self, input_dim, hidden_dim,
                 lin1=None, mask_encoding=None, output_bias=None):
        super(AutoRegressiveNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_bias = output_bias

        if mask_encoding is None:
            # the dependency structure is chosen at random
            self.mask_encoding = 1 + torch.multinomial(torch.ones(input_dim - 1) / (input_dim - 1),
                                                       num_samples=input_dim, replacement=True)
        else:
            # the dependency structure is given by the user (probably just passing on an
            # encoding_mask from a previously initialized AutoRegressiveNN)
            self.mask_encoding = mask_encoding

        self.mask1 = Variable(torch.zeros(hidden_dim, input_dim))
        self.mask2 = Variable(torch.zeros(input_dim, hidden_dim))
        for k in range(input_dim):
            m_k = self.mask_encoding[k]
            self.mask1[k, 0:m_k] = torch.ones(m_k)
            self.mask2[m_k:input_dim, k] = torch.ones(input_dim - m_k)

        if lin1 is None:
            self.lin1 = MaskedLinear(input_dim, hidden_dim, self.mask1)
        else:
            self.lin1 = lin1

        self.lin2 = MaskedLinear(hidden_dim, input_dim, self.mask2)
        self.relu = nn.ReLU()

    def get_lin1(self):
        return self.lin1

    def get_mask_encoding(self):
        # basically the quantity m(k) in the MADE paper
        return self.mask_encoding

    def forward(self, z):
        h = self.relu(self.lin1(z))
        out = self.lin2(h)
        if self.output_bias is not None:
            return out + self.output_bias * ng_ones(out.size())
        else:
            return out
