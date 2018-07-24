from __future__ import absolute_import, division, print_function

import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

def sample_mask_indices(input_dimension, hidden_dimension, simple=False, conditional=True):
    """
    Samples the indices assigned to hidden units during the construction of MADE masks

    :param input_dimension: the dimensionality of the input variable
    :type input_dimension: int
    :param hidden_dimension: the dimensionality of the hidden layer
    :type hidden_dimension: int
    :param simple: True to sample indices uniformly, false to space indices evenly and round up or down randomly
    :type simple: bool
    """
    start_integer = 0 if conditional else 1

    if simple:
        return np.random.randint(start_integer, input_dimension, size=(hidden_dimension,))
    else:
        mk = np.linspace(start_integer, input_dimension-1, hidden_dimension)
        ints = np.array(mk, dtype=int)

        # NOTE: Maybe we'd prefer a vector of rand here?
        ints += (np.random.rand() < mk - ints)

        return ints

def create_mask(input_dimension, observed_dimension, hidden_dimension, num_layers, permutation, output_dim_multiplier):
    """
    Creates MADE masks for a conditional distribution

    :param input_dimension: the dimensionality of the input variable
    :type input_dimension: int
    :param observed_dimension: the dimensionality of the variable that is conditioned on (for conditional densities)
    :type observed_dimension: int
    :param hidden_dimension: the dimensionality of the hidden layer(s)
    :type hidden_dimension: int
    :param num_layers: the number of hidden layers for which to create masks
    :type num_layers: int
    :param permutation: the order of the input variables
    :type permutation: np.array
    :param output_dim_multiplier: tiles the output (e.g. for when a separate mean and scale parameter are desired)
    :type output_dim_multiplier: int
    """
    # Create mask indices for input, hidden layers, and final layer
    # We use 0 to refer to the elements of the variable being conditioned on, and range(1:(D_latent+1)) for the input variable
    m_input = np.concatenate((np.zeros(observed_dimension), 1+permutation))
    m_w = [sample_mask_indices(input_dimension, hidden_dimension, conditional=observed_dimension>0) for i in range(num_layers)]
    m_v = np.tile(permutation, output_dim_multiplier)

    # Create mask from input to output for the skips connections
    M_A = (1.0*(np.atleast_2d(m_v).T >= np.atleast_2d(m_input)))

    # Create mask from input to first hidden layer, and between subsequent hidden layers
    M_W = [(1.0*(np.atleast_2d(m_w[0]).T >= np.atleast_2d(m_input)))]
    for i in range(1, num_layers):
        M_W.append(1.0*(np.atleast_2d(m_w[i]).T >= np.atleast_2d(m_w[i-1])))

    # Create mask from last hidden layer to output layer
    M_V = (1.0*(np.atleast_2d(m_v).T >= np.atleast_2d(m_w[-1])))

    return M_W, M_V, M_A

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
    An implementation of a MADE-like auto-regressive neural network.

    Reference:
    MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle

    :param input_dimension: the dimensionality of the input
    :type input_dimension: int
    :param hidden_dimension: the dimensionality of the hidden units
    :type hidden_dimension: int
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
        is such that the Jacobian is upper triangular. By default this is the identity permutation.
    :type permutation: np.array
    """

    def __init__(self, input_dim, hidden_dim, output_dim_multiplier=1,
                 permutation=None, skip_connections=False, num_layers=1, nonlinearity=nn.ReLU()):
        super(AutoRegressiveNN, self).__init__()
        if input_dim == 1:
            warnings.warn('AutoRegressiveNN input_dim = 1. Consider using an affine transformation instead.')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim_multiplier = output_dim_multiplier
        self.num_layers = num_layers

        if permutation is None:
            # order the variables in the order that they're given
            self.permutation = np.arange(input_dim)
        else:
            # the permutation is chosen by the user
            self.permutation = permutation

        # Create masks
        M_W, M_V, M_A = create_mask(input_dimension=input_dim, observed_dimension=0, hidden_dimension=hidden_dim, num_layers=num_layers, permutation=self.permutation, output_dim_multiplier=output_dim_multiplier)
        self.M_W = [torch.FloatTensor(M) for M in M_W]
        self.M_V = torch.FloatTensor(M_V)

        # Create masked layers
        layers = [MaskedLinear(input_dim, hidden_dim, self.M_W[0])]
        for i in range(1, num_layers):
            layers.append(MaskedLinear(hidden_dim, hidden_dim, self.M_W[i]))
        self.layers = nn.ModuleList(layers)

        if skip_connections:
          self.M_A = torch.FloatTensor(M_A)
          self.skip_p = MaskedLinear(input_dim, input_dim*output_dim_multiplier, self.M_A, bias=False)
        else:
          self.skip_p = None
        self.p = MaskedLinear(hidden_dim, input_dim*output_dim_multiplier, self.M_V)

        # Save the nonlinearity
        self.f = nonlinearity

    def get_permutation(self):
        """
        Get the permutation applied to the inputs (by default this is chosen at random)
        """
        return self.permutation

    def forward(self, x):
        """
        the forward method
        """
        h = x
        for layer in self.layers:
          h = self.f(layer(h))

        if self.skip_p is not None:
          h = self.p(h) + self.skip_p(x)
        else:
          h = self.p(h)

        return h