# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Transform, constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.nn import DenseNN


@copy_docs_from(Transform)
class ConditionedPlanar(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, params):
        super().__init__(cache_size=1)
        self._params = params
        self._cached_logDetJ = None

    # This method ensures that torch(u_hat, w) > -1, required for invertibility
    def u_hat(self, u, w):
        alpha = torch.matmul(u.unsqueeze(-2), w.unsqueeze(-1)).squeeze(-1)
        a_prime = -1 + F.softplus(alpha)
        return u + (a_prime - alpha) * w.div(w.pow(2).sum(dim=-1, keepdim=True))

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x => y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        bias, u, w = self._params() if callable(self._params) else self._params

        # x ~ (batch_size, dim_size, 1)
        # w ~ (batch_size, 1, dim_size)
        # bias ~ (batch_size, 1)
        act = torch.tanh(torch.matmul(w.unsqueeze(-2), x.unsqueeze(-1)).squeeze(-1) + bias)
        u_hat = self.u_hat(u, w)
        y = x + u_hat * act

        psi_z = (1. - act.pow(2)) * w
        self._cached_logDetJ = torch.log(
            torch.abs(1 + torch.matmul(psi_z.unsqueeze(-2), u_hat.unsqueeze(-1)).squeeze(-1).squeeze(-1)))

        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. As noted above, this implementation is incapable of
        inverting arbitrary values `y`; rather it assumes `y` is the result of a
        previously computed application of the bijector to some `x` (which was
        cached on the forward call)
        """

        raise KeyError("ConditionedPlanar object expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cached_logDetJ


@copy_docs_from(ConditionedPlanar)
class Planar(ConditionedPlanar, TransformModule):
    r"""
    A 'planar' bijective transform with equation,

        :math:`\mathbf{y} = \mathbf{x} + \mathbf{u}\tanh(\mathbf{w}^T\mathbf{z}+b)`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    and the learnable parameters are :math:`b\in\mathbb{R}`,
    :math:`\mathbf{u}\in\mathbb{R}^D`, :math:`\mathbf{w}\in\mathbb{R}^D` for
    input dimension :math:`D`. For this to be an invertible transformation, the
    condition :math:`\mathbf{w}^T\mathbf{u}>-1` is enforced.

    Together with :class:`~pyro.distributions.TransformedDistribution` this provides
    a way to create richer variational approximations.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = Planar(10)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse of this transform does not possess an analytical solution and is
    left unimplemented. However, the inverse is cached when the forward operation is
    called during sampling, and so samples drawn using the planar transform can be
    scored.

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int

    References:

    [1] Danilo Jimenez Rezende, Shakir Mohamed. Variational Inference with
    Normalizing Flows. [arXiv:1505.05770]

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, input_dim):
        super().__init__(self._params)

        self.bias = nn.Parameter(torch.Tensor(1,))
        self.u = nn.Parameter(torch.Tensor(input_dim,))
        self.w = nn.Parameter(torch.Tensor(input_dim,))

        self.input_dim = input_dim
        self.reset_parameters()

    def _params(self):
        return self.bias, self.u, self.w

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u.size(0))
        self.w.data.uniform_(-stdv, stdv)
        self.u.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


@copy_docs_from(ConditionalTransformModule)
class ConditionalPlanar(ConditionalTransformModule):
    r"""
    A conditional 'planar' bijective transform using the equation,

        :math:`\mathbf{y} = \mathbf{x} + \mathbf{u}\tanh(\mathbf{w}^T\mathbf{z}+b)`

    where :math:`\mathbf{x}` are the inputs with dimension :math:`D`,
    :math:`\mathbf{y}` are the outputs, and the pseudo-parameters
    :math:`b\in\mathbb{R}`, :math:`\mathbf{u}\in\mathbb{R}^D`, and
    :math:`\mathbf{w}\in\mathbb{R}^D` are the output of a function, e.g. a NN,
    with input :math:`z\in\mathbb{R}^{M}` representing the context variable to
    condition on. For this to be an invertible transformation, the condition
    :math:`\mathbf{w}^T\mathbf{u}>-1` is enforced.

    Together with :class:`~pyro.distributions.ConditionalTransformedDistribution`
    this provides a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn.dense_nn import DenseNN
    >>> input_dim = 10
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [1, input_dim, input_dim]
    >>> hypernet = DenseNN(context_dim, [50, 50], param_dims)
    >>> transform = ConditionalPlanar(hypernet)
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size])) # doctest: +SKIP

    The inverse of this transform does not possess an analytical solution and is
    left unimplemented. However, the inverse is cached when the forward operation is
    called during sampling, and so samples drawn using the planar transform can be
    scored.

    :param nn: a function inputting the context variable and outputting a triplet of
        real-valued parameters of dimensions :math:`(1, D, D)`.
    :type nn: callable

    References:
    [1] Variational Inference with Normalizing Flows [arXiv:1505.05770]
    Danilo Jimenez Rezende, Shakir Mohamed

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def _params(self, context):
        return self.nn(context)

    def condition(self, context):
        params = partial(self._params, context)
        return ConditionedPlanar(params)


def planar(input_dim):
    """
    A helper function to create a :class:`~pyro.distributions.transforms.Planar`
    object for consistency with other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int

    """

    return Planar(input_dim)


def conditional_planar(input_dim, context_dim, hidden_dims=None):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.ConditionalPlanar` object that takes care
    of constructing a dense network with the correct input/output dimensions.

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
    nn = DenseNN(context_dim, hidden_dims, param_dims=[1, input_dim, input_dim])
    return ConditionalPlanar(nn)
