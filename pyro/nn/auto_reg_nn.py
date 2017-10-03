import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import Parameter
from pyro.util import ng_ones


class MaskedLinear(nn.Linear):
    """
    a linear mapping with a given mask on the weights (arbitrary bias)
    """

    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)
        #self.mask = mask

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

    def __init__(self, input_dim, hidden_dim, output_dim_multiplier=1,
                 mask_encoding=None, output_bias=None, permutation=None):
        super(AutoRegressiveNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_bias = output_bias
        self.output_dim_multiplier = output_dim_multiplier

        if mask_encoding is None:
            # the dependency structure is chosen at random
            self.mask_encoding = 1 + torch.multinomial(torch.ones(input_dim - 1) / (input_dim - 1),
                                                       num_samples=hidden_dim, replacement=True)
        else:
            # the dependency structure is given by the user
            self.mask_encoding = mask_encoding

        if permutation is None:
            self.permutation = torch.randperm(input_dim)
        else:
            self.permutation = permutation

        self.mask1 = Variable(torch.zeros(hidden_dim, input_dim))
        self.mask2 = Variable(torch.zeros(input_dim * self.output_dim_multiplier, hidden_dim))
        #mask1 = Variable(torch.zeros(hidden_dim, input_dim))
        #mask2 = Variable(torch.zeros(input_dim * self.output_dim_multiplier, hidden_dim))
        #self.register_buffer('mask1', mask1)
        #self.register_buffer('mask2', mask2)

        for k in range(hidden_dim):
            # fill in mask1
            m_k = self.mask_encoding[k]
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
        # basically the quantity m(k) in the MADE paper
        return self.mask_encoding

    def get_permutation(self):
        return self.permutation

    def forward(self, z):
        h = self.relu(self.lin1(z))
        out = self.lin2(h)
        if self.output_bias is not None:
            return out + self.output_bias * ng_ones(out.size())
        else:
            return out
