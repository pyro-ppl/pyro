# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import constraints
from torch.distributions.transforms import Transform, SigmoidTransform
import torch.nn.functional as F

from pyro.distributions.util import copy_docs_from
from pyro.nn import AutoRegressiveNN

eps = 1e-8


class ELUTransform(Transform):
    r"""
    Bijective transform via the mapping :math:`y = \text{ELU}(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, ELUTransform)

    def _call(self, x):
        return F.elu(x)

    def _inverse(self, y):
        return torch.max(y, torch.zeros_like(y)) + torch.min(torch.log1p(y + eps), torch.zeros_like(y))

    def log_abs_det_jacobian(self, x, y):
        return -F.relu(-x)


def elu():
    """
    A helper function to create an
    :class:`~pyro.distributions.transform.ELUTransform` object for consistency with
    other helpers.
    """
    return ELUTransform()


class LeakyReLUTransform(Transform):
    r"""
    Bijective transform via the mapping :math:`y = \text{LeakyReLU}(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, LeakyReLUTransform)

    def _call(self, x):
        return F.leaky_relu(x)

    def _inverse(self, y):
        return F.leaky_relu(y, negative_slope=100.0)

    def log_abs_det_jacobian(self, x, y):
        return torch.where(x >= 0., torch.zeros_like(x), torch.ones_like(x) * math.log(0.01))


def leaky_relu():
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.LeakyReLUTransform` object for
    consistency with other helpers.
    """
    return LeakyReLUTransform()


class TanhTransform(Transform):
    r"""
    Bijective transform via the mapping :math:`y = \text{tanh}(x)`.
    """
    domain = constraints.real
    codomain = constraints.interval(-1., 1.)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        eps = torch.finfo(y.dtype).eps
        return self.atanh(y.clamp(min=-1. + eps, max=1. - eps))

    def log_abs_det_jacobian(self, x, y):
        return - 2. * (x - math.log(2.) + F.softplus(- 2. * x))


def tanh():
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.TanhTransform` object for consistency
    with other helpers.
    """
    return TanhTransform()


@copy_docs_from(TransformModule)
class NeuralAutoregressive(TransformModule):
    """
    An implementation of the deep Neural Autoregressive Flow (NAF) bijective
    transform of the "IAF flavour" that can be used for sampling and scoring samples
    drawn from it (but not arbitrary ones).

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> arn = AutoRegressiveNN(10, [40], param_dims=[16]*3)
    >>> transform = NeuralAutoregressive(arn, hidden_units=16)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse operation is not implemented. This would require numerical
    inversion, e.g., using a root finding method - a possibility for a future
    implementation.

    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns a tuple of three real-valued tensors, whose last dimension is the
        input dimension, and whose penultimate dimension is equal to hidden_units.
    :type autoregressive_nn: nn.Module
    :param hidden_units: the number of hidden units to use in the NAF transformation
        (see Eq (8) in reference)
    :type hidden_units: int
    :param activation: Activation function to use. One of 'ELU', 'LeakyReLU',
        'sigmoid', or 'tanh'.
    :type activation: string

    Reference:

    [1] Chin-Wei Huang, David Krueger, Alexandre Lacoste, Aaron Courville. Neural
    Autoregressive Flows. [arXiv:1804.00779]


    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, autoregressive_nn, hidden_units=16, activation='sigmoid'):
        super().__init__(cache_size=1)

        # Create the intermediate transform used
        name_to_mixin = {
            'ELU': ELUTransform,
            'LeakyReLU': LeakyReLUTransform,
            'sigmoid': SigmoidTransform,
            'tanh': TanhTransform}
        if activation not in name_to_mixin:
            raise ValueError('Invalid activation function "{}"'.format(activation))
        self.T = name_to_mixin[activation]()

        self.arn = autoregressive_nn
        self.hidden_units = hidden_units
        self.logsoftmax = nn.LogSoftmax(dim=-2)
        self._cached_log_df_inv_dx = None
        self._cached_A = None
        self._cached_W_pre = None
        self._cached_C = None
        self._cached_T_C = None

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        # A, W, b ~ batch_shape x hidden_units x event_shape
        A, W_pre, b = self.arn(x)
        T = self.T

        # Divide the autoregressive output into the component activations
        A = F.softplus(A)
        C = A * x.unsqueeze(-2) + b
        W = F.softmax(W_pre, dim=-2)
        T_C = T(C)
        D = (W * T_C).sum(dim=-2)
        y = T.inv(D)

        self._cached_log_df_inv_dx = T.inv.log_abs_det_jacobian(D, y)
        self._cached_A = A
        self._cached_W_pre = W_pre
        self._cached_C = C
        self._cached_T_C = T_C

        return y

    # This method returns log(abs(det(dy/dx)), which is equal to -log(abs(det(dx/dy))
    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """

        A = self._cached_A
        W_pre = self._cached_W_pre
        C = self._cached_C
        T_C = self._cached_T_C
        T = self.T

        log_dydD = self._cached_log_df_inv_dx
        log_dDdx = torch.logsumexp(torch.log(A + eps) + self.logsoftmax(W_pre) +
                                   T.log_abs_det_jacobian(C, T_C), dim=-2)
        log_det = log_dydD + log_dDdx
        return log_det.sum(-1)


def neural_autoregressive(input_dim, hidden_dims=None, activation='sigmoid', width=16):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.NeuralAutoregressive` object that takes
    care of constructing an autoregressive network with the correct input/output
    dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param hidden_dims: The desired hidden dimensions of the autoregressive network.
        Defaults to using [3*input_dim + 1]
    :type hidden_dims: list[int]
    :param activation: Activation function to use. One of 'ELU', 'LeakyReLU',
        'sigmoid', or 'tanh'.
    :type activation: string
    :param width: The width of the "multilayer perceptron" in the transform (see
        paper). Defaults to 16
    :type width: int

    """

    if hidden_dims is None:
        hidden_dims = [3 * input_dim + 1]
    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=[width] * 3)
    return NeuralAutoregressive(arn, hidden_units=width, activation=activation)
