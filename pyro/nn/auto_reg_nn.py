from __future__ import absolute_import, division, print_function

import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F


def sample_mask_indices(input_dim, hidden_dim, simple=True):
    """
    Samples the indices assigned to hidden units during the construction of MADE masks

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param hidden_dim: the dimensionality of the hidden layer
    :type hidden_dim: int
    :param simple: True to space fractional indices by rounding to nearest int, false round randomly
    :type simple: bool
    """
    indices = torch.linspace(1, input_dim, steps=hidden_dim, device='cpu').to(torch.Tensor().device)
    if simple:
        # Simple procedure tries to space fractional indices evenly by rounding to nearest int
        return torch.round(indices)
    else:
        # "Non-simple" procedure creates fractional indices evenly then rounds at random
        ints = indices.floor()
        ints += torch.bernoulli(indices - ints)
        return ints


def create_mask(input_dim, observed_dim, hidden_dims, permutation, output_dim_multiplier):
    """
    Creates MADE masks for a conditional distribution

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param observed_dim: the dimensionality of the variable that is conditioned on (for conditional densities)
    :type observed_dim: int
    :param hidden_dims: the dimensionality of the hidden layers(s)
    :type hidden_dims: list[int]
    :param permutation: the order of the input variables
    :type permutation: torch.LongTensor
    :param output_dim_multiplier: tiles the output (e.g. for when a separate mean and scale parameter are desired)
    :type output_dim_multiplier: int
    """
    # Create mask indices for input, hidden layers, and final layer
    # We use 0 to refer to the elements of the variable being conditioned on,
    # and range(1:(D_latent+1)) for the input variable
    var_index = torch.empty(permutation.shape, dtype=torch.get_default_dtype())
    var_index[permutation] = torch.arange(input_dim, dtype=torch.get_default_dtype())

    # Create the indices that are assigned to the neurons
    input_indices = torch.cat((torch.zeros(observed_dim), 1 + var_index))
    hidden_indices = [sample_mask_indices(input_dim, h) for h in hidden_dims]
    output_indices = (var_index + 1).repeat(output_dim_multiplier)

    # Create mask from input to output for the skips connections
    mask_skip = (output_indices.unsqueeze(-1) > input_indices.unsqueeze(0)).type_as(var_index)

    # Create mask from input to first hidden layer, and between subsequent hidden layers
    # NOTE: The masks created follow a slightly different pattern than that given in Germain et al. Figure 1
    # The output first in the order (e.g. x_2 in the figure) is connected to hidden units rather than being unattached
    # Tracing a path back through the network, however, this variable will still be unconnected to any input variables
    masks = [(hidden_indices[0].unsqueeze(-1) > input_indices.unsqueeze(0)).type_as(var_index)]
    for i in range(1, len(hidden_dims)):
        masks.append((hidden_indices[i].unsqueeze(-1) >= hidden_indices[i - 1].unsqueeze(0)).type_as(var_index))

    # Create mask from last hidden layer to output layer
    masks.append((output_indices.unsqueeze(-1) >= hidden_indices[-1].unsqueeze(0)).type_as(var_index))

    return masks, mask_skip


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

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param count_params: number of parameters to return from a forward call of the network.
    :type count_params: int
    :param param_dim: the dimensionality of each parameter
    :type param_dim: int
    :param output_dim_multiplier: the dimensionality of the output is given by input_dim x output_dim_multiplier.
        specifically the shape of the output for a single vector input is [output_dim_multiplier, input_dim].
        for any i, j in range(0, output_dim_multiplier) the subset of outputs [i, :] has identical
        autoregressive structure to [j, :]. defaults to `1`
    :type output_dim_multiplier: int
    :param permutation: an optional permutation that is applied to the inputs and controls the order of the
        autoregressive factorization. in particular for the identity permutation the autoregressive structure
        is such that the Jacobian is upper triangular. By default this is chosen at random.
    :type permutation: torch.LongTensor
    :param skip_connections: Whether to add skip connections from the input to the output.
    :type skip_connections: bool
    :param nonlinearity: The nonlinearity to use in the feedforward network.
    :type nonlinearity: torch.nn.module
    """

    def __init__(
            self,
            input_dim,
            hidden_dims,
            count_params=2,
            param_dim=1,
            permutation=None,
            skip_connections=False,
            nonlinearity=nn.ReLU()):
        super(AutoRegressiveNN, self).__init__()
        if input_dim == 1:
            warnings.warn('AutoRegressiveNN input_dim = 1. Consider using an affine transformation instead.')
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.count_params = count_params
        self.param_dim = param_dim

        # Hidden dimension must be not less than the input otherwise it isn't
        # possible to connect to the outputs correctly
        for h in hidden_dims:
            if h < input_dim:
                raise ValueError('Hidden dimension must not be less than input dimension.')

        if permutation is None:
            # By default set a random permutation of variables, which is important for performance with multiple steps
            self.permutation = torch.randperm(input_dim, device='cpu').to(torch.Tensor().device)
        else:
            # The permutation is chosen by the user
            self.permutation = permutation.type(dtype=torch.int64)

        # Create masks
        output_dim_multiplier = count_params * param_dim
        self.masks, self.mask_skip = create_mask(
            input_dim=input_dim, observed_dim=0, hidden_dims=hidden_dims, permutation=self.permutation,
            output_dim_multiplier=output_dim_multiplier)

        # Create masked layers
        layers = [MaskedLinear(input_dim, hidden_dims[0], self.masks[0])]
        for i in range(1, len(hidden_dims)):
            layers.append(MaskedLinear(hidden_dims[i - 1], hidden_dims[i], self.masks[i]))
        layers.append(MaskedLinear(hidden_dims[-1], input_dim * output_dim_multiplier, self.masks[-1]))
        self.layers = nn.ModuleList(layers)

        if skip_connections:
            self.skip_layer = MaskedLinear(input_dim, input_dim * output_dim_multiplier, self.mask_skip, bias=False)
        else:
            self.skip_layer = None

        # Save the nonlinearity
        self.f = nonlinearity

    def get_permutation(self):
        """
        Get the permutation applied to the inputs (by default this is chosen at random)
        """
        return self.permutation

    def forward(self, x):
        """
        The forward method
        """
        h = x
        for layer in self.layers[:-1]:
            h = self.f(layer(h))
        h = self.layers[-1](h)

        if self.skip_layer is not None:
            h = h + self.skip_layer(x)

        # Shape the output into self.count_params of dimension self.param_dim*self.input_dim
        if self.count_params == 1:
            return h
        else:
            params = [h[..., (i * self.param_dim * self.input_dim):((i + 1) * self.param_dim * self.input_dim)]
                      for i in range(self.count_params)]
            if self.param_dim > 1:
                # NOTE: Better way to do this?
                if len(x.size()) > 1:
                    params = [p.reshape(-1, self.param_dim, self.input_dim) for p in params]
                else:
                    params = [p.reshape(self.param_dim, self.input_dim) for p in params]

            return tuple(params)
