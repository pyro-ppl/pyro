from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class DenseNN(nn.Module):
    """
    An implementation of a simple dense feedforward network, for use in, e.g., some conditional flows such as
    :class:`pyro.distributions.transforms.ConditionalPlanarFlow`.

    # *** TODO: Fix up example usage, params, etc. ***
    Example usage:

    >>> x = torch.randn(100, 10)
    >>> arn = AutoRegressiveNN(10, [50], param_dims=[1])
    >>> p = arn(x)  # 1 parameters of size (100, 10)
    >>> arn = AutoRegressiveNN(10, [50], param_dims=[1, 1])
    >>> m, s = arn(x) # 2 parameters of size (100, 10)
    >>> arn = AutoRegressiveNN(10, [50], param_dims=[1, 5, 3])
    >>> a, b, c = arn(x) # 3 parameters of sizes, (100, 1, 10), (100, 5, 10), (100, 3, 10)

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n, input_dim) for p_n in param_dims
        when p_n > 1 and dimension (input_dim) when p_n == 1. The default is [1, 1], i.e. output two parameters
        of dimension (input_dim), which is useful for inverse autoregressive flow.
    :type param_dims: list[int]
    :param permutation: an optional permutation that is applied to the inputs and controls the order of the
        autoregressive factorization. in particular for the identity permutation the autoregressive structure
        is such that the Jacobian is upper triangular. By default this is chosen at random.
    :type permutation: torch.LongTensor
    :param skip_connections: Whether to add skip connections from the input to the output.
    :type skip_connections: bool
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.module

    Reference:

    MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle

    """

    def __init__(
            self,
            input_dim,
            hidden_dims,
            param_dims=[1, 1],
            nonlinearity=nn.ReLU()):
        super(DenseNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)
        self.all_ones = (torch.tensor(param_dims) == 1).all().item()

        # Calculate the indices on the output corresponding to each parameter
        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        # Create masked layers
        layers = [nn.Linear(input_dim, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        layers.append(nn.Linear(hidden_dims[-1], input_dim * self.output_multiplier))
        self.layers = nn.ModuleList(layers)

        # Save the nonlinearity
        self.f = nonlinearity

    def forward(self, x):
        """
        The forward method
        """
        h = x
        for layer in self.layers[:-1]:
            h = self.f(layer(h))
        h = self.layers[-1](h)

        # Shape the output, squeezing the parameter dimension if all ones
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier, self.input_dim])

            # Squeeze dimension if all parameters are one dimensional
            if self.count_params == 1:
                return h

            elif self.all_ones:
                return torch.unbind(h, dim=-2)

            # If not all ones, then probably don't want to squeeze a single dimension parameter
            else:
                return tuple([h[..., s, :] for s in self.param_slices])
