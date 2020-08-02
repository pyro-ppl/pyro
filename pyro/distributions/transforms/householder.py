# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import Transform, constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.nn import DenseNN


class ConditionedHouseholder(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1
    volume_preserving = True

    def __init__(self, u_unnormed=None):
        super().__init__(cache_size=1)
        self.u_unnormed = u_unnormed

    # Construct normalized vectors for Householder transform
    def u(self):
        u_unnormed = self.u_unnormed() if callable(self.u_unnormed) else self.u_unnormed
        norm = torch.norm(u_unnormed, p=2, dim=-1, keepdim=True)
        return torch.div(u_unnormed, norm)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        y = x
        u = self.u()
        for idx in range(u.size(-2)):
            projection = (u[..., idx, :] * y).sum(dim=-1, keepdim=True) * u[..., idx, :]
            y = y - 2. * projection
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. The Householder transformation, H, is "involutory," i.e.
        H^2 = I. If you reflect a point around a plane, then the same operation will
        reflect it back
        """

        x = y
        u = self.u()
        for jdx in reversed(range(u.size(-2))):
            # NOTE: Need to apply transforms in reverse order from forward operation!
            projection = (u[..., jdx, :] * x).sum(dim=-1, keepdim=True) * u[..., jdx, :]
            x = x - 2. * projection
        return x

    def log_abs_det_jacobian(self, x, y):
        r"""
        Calculates the elementwise determinant of the log jacobian. Householder flow
        is measure preserving, so :math:`\log(|detJ|) = 0`
        """

        return torch.zeros(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device)


@copy_docs_from(TransformModule)
class Householder(ConditionedHouseholder, TransformModule):
    r"""
    Represents multiple applications of the Householder bijective transformation. A
    single Householder transformation takes the form,

        :math:`\mathbf{y} = (I - 2*\frac{\mathbf{u}\mathbf{u}^T}{||\mathbf{u}||^2})\mathbf{x}`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    and the learnable parameters are :math:`\mathbf{u}\in\mathbb{R}^D` for input
    dimension :math:`D`.

    The transformation represents the reflection of :math:`\mathbf{x}` through the
    plane passing through the origin with normal :math:`\mathbf{u}`.

    :math:`D` applications of this transformation are able to transform standard
    i.i.d. standard Gaussian noise into a Gaussian variable with an arbitrary
    covariance matrix. With :math:`K<D` transformations, one is able to approximate
    a full-rank Gaussian distribution using a linear transformation of rank
    :math:`K`.

    Together with :class:`~pyro.distributions.TransformedDistribution` this provides
    a way to create richer variational approximations.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = Householder(10, count_transforms=5)
    >>> pyro.module("my_transform", p) # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int
    :param count_transforms: number of applications of Householder transformation to
        apply.
    :type count_transforms: int

    References:

    [1] Jakub M. Tomczak, Max Welling. Improving Variational Auto-Encoders using
    Householder Flow. [arXiv:1611.09630]

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1
    volume_preserving = True

    def __init__(self, input_dim, count_transforms=1):
        super().__init__()

        self.input_dim = input_dim
        if count_transforms < 1:
            raise ValueError('Number of Householder transforms, {}, is less than 1!'.format(count_transforms))
        elif count_transforms > input_dim:
            warnings.warn(
                "Number of Householder transforms, {}, is greater than input dimension {}, which is an \
over-parametrization!".format(count_transforms, input_dim))
        self.u_unnormed = nn.Parameter(torch.Tensor(count_transforms, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u_unnormed.size(-1))
        self.u_unnormed.data.uniform_(-stdv, stdv)


@copy_docs_from(ConditionalTransformModule)
class ConditionalHouseholder(ConditionalTransformModule):
    r"""
    Represents multiple applications of the Householder bijective transformation
    conditioning on an additional context. A single Householder transformation takes
    the form,

        :math:`\mathbf{y} = (I - 2*\frac{\mathbf{u}\mathbf{u}^T}{||\mathbf{u}||^2})\mathbf{x}`

    where :math:`\mathbf{x}` are the inputs with dimension :math:`D`,
    :math:`\mathbf{y}` are the outputs, and :math:`\mathbf{u}\in\mathbb{R}^D`
    is the output of a function, e.g. a NN, with input :math:`z\in\mathbb{R}^{M}`
    representing the context variable to condition on.

    The transformation represents the reflection of :math:`\mathbf{x}` through the
    plane passing through the origin with normal :math:`\mathbf{u}`.

    :math:`D` applications of this transformation are able to transform standard
    i.i.d. standard Gaussian noise into a Gaussian variable with an arbitrary
    covariance matrix. With :math:`K<D` transformations, one is able to approximate
    a full-rank Gaussian distribution using a linear transformation of rank
    :math:`K`.

    Together with :class:`~pyro.distributions.ConditionalTransformedDistribution`
    this provides a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn.dense_nn import DenseNN
    >>> input_dim = 10
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [input_dim]
    >>> hypernet = DenseNN(context_dim, [50, 50], param_dims)
    >>> transform = ConditionalHouseholder(input_dim, hypernet)
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size])) # doctest: +SKIP

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int
    :param nn: a function inputting the context variable and outputting a triplet of
        real-valued parameters of dimensions :math:`(1, D, D)`.
    :type nn: callable
    :param count_transforms: number of applications of Householder transformation to
        apply.
    :type count_transforms: int

    References:

    [1] Jakub M. Tomczak, Max Welling. Improving Variational Auto-Encoders using
    Householder Flow. [arXiv:1611.09630]

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, input_dim, nn, count_transforms=1):
        super().__init__()
        self.nn = nn
        self.input_dim = input_dim
        if count_transforms < 1:
            raise ValueError('Number of Householder transforms, {}, is less than 1!'.format(count_transforms))
        elif count_transforms > input_dim:
            warnings.warn(
                "Number of Householder transforms, {}, is greater than input dimension {}, which is an \
over-parametrization!".format(count_transforms, input_dim))
        self.count_transforms = count_transforms

    def _u_unnormed(self, context):
        # u_unnormed ~ (count_transforms, input_dim)
        # Hence, input_dim must divide
        u_unnormed = self.nn(context)
        if self.count_transforms == 1:
            u_unnormed = u_unnormed.unsqueeze(-2)
        else:
            u_unnormed = torch.stack(u_unnormed, dim=-2)
        return u_unnormed

    def condition(self, context):
        u_unnormed = partial(self._u_unnormed, context)
        return ConditionedHouseholder(u_unnormed)


def householder(input_dim, count_transforms=None):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.Householder` object for consistency with
    other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param count_transforms: number of applications of Householder transformation to
        apply.
    :type count_transforms: int

    """

    if count_transforms is None:
        count_transforms = input_dim // 2 + 1
    return Householder(input_dim, count_transforms=count_transforms)


def conditional_householder(input_dim, context_dim, hidden_dims=None, count_transforms=1):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.ConditionalHouseholder` object that takes
    care of constructing a dense network with the correct input/output dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param context_dim: Dimension of context variable
    :type context_dim: int
    :param hidden_dims: The desired hidden dimensions of the dense network. Defaults
        to using [input_dim * 10, input_dim * 10]
    :type hidden_dims: list[int]

    """

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    nn = DenseNN(context_dim, hidden_dims, param_dims=[input_dim] * count_transforms)
    return ConditionalHouseholder(input_dim, nn, count_transforms)
