# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


class ConditionalDenseNN(torch.nn.Module):
    """
    An implementation of a simple dense feedforward network taking a context variable, for use in, e.g.,
    some conditional flows such as :class:`pyro.distributions.transforms.ConditionalAffineCoupling`.

    Example usage:

    >>> input_dim = 10
    >>> context_dim = 5
    >>> x = torch.rand(100, input_dim)
    >>> z = torch.rand(100, context_dim)
    >>> nn = ConditionalDenseNN(input_dim, context_dim, [50], param_dims=[1, input_dim, input_dim])
    >>> a, b, c = nn(x, context=z)  # parameters of size (100, 1), (100, 10), (100, 10)

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param context_dim: the dimensionality of the context variable
    :type context_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n,) for p_n in param_dims
        when p_n > 1 and dimension () when p_n == 1. The default is [1, 1], i.e. output two parameters of dimension ().
    :type param_dims: list[int]
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.Module

    """

    def __init__(
            self,
            input_dim,
            context_dim,
            hidden_dims,
            param_dims=[1, 1],
            nonlinearity=torch.nn.ReLU()):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)

        # Calculate the indices on the output corresponding to each parameter
        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        # Create masked layers
        layers = [torch.nn.Linear(input_dim+context_dim, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers.append(torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        layers.append(torch.nn.Linear(hidden_dims[-1], self.output_multiplier))
        self.layers = torch.nn.ModuleList(layers)

        # Save the nonlinearity
        self.f = nonlinearity

    def forward(self, x, context):
        # We must be able to broadcast the size of the context over the input
        context = context.expand(x.size()[:-1]+(context.size(-1),))

        x = torch.cat([context, x], dim=-1)
        return self._forward(x)

    def _forward(self, x):
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
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier])

            if self.count_params == 1:
                return h

            else:
                return tuple([h[..., s] for s in self.param_slices])


class DenseNN(ConditionalDenseNN):
    """
    An implementation of a simple dense feedforward network, for use in, e.g., some conditional flows such as
    :class:`pyro.distributions.transforms.ConditionalPlanarFlow` and other unconditional flows such as
    :class:`pyro.distributions.transforms.AffineCoupling` that do not require an autoregressive network.

    Example usage:

    >>> input_dim = 10
    >>> context_dim = 5
    >>> z = torch.rand(100, context_dim)
    >>> nn = DenseNN(context_dim, [50], param_dims=[1, input_dim, input_dim])
    >>> a, b, c = nn(z)  # parameters of size (100, 1), (100, 10), (100, 10)

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n,) for p_n in param_dims
        when p_n > 1 and dimension () when p_n == 1. The default is [1, 1], i.e. output two parameters of dimension ().
    :type param_dims: list[int]
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.module

    """

    def __init__(
            self,
            input_dim,
            hidden_dims,
            param_dims=[1, 1],
            nonlinearity=torch.nn.ReLU()):
        super(DenseNN, self).__init__(
            input_dim,
            0,
            hidden_dims,
            param_dims=param_dims,
            nonlinearity=nonlinearity
        )

    def forward(self, x):
        return self._forward(x)
