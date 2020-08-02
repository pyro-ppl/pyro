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
class ConditionedRadial(Transform):
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

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from the base distribution (or the output
        of a previous transform)
        """
        x0, alpha_prime, beta_prime = self._params() if callable(self._params) else self._params

        # Ensure invertibility using approach in appendix A.2
        alpha = F.softplus(alpha_prime)
        beta = -alpha + F.softplus(beta_prime)

        # Compute y and logDet using Equation 14.
        diff = x - x0
        r = diff.norm(dim=-1, keepdim=True)
        h = (alpha + r).reciprocal()
        h_prime = - (h ** 2)
        beta_h = beta * h

        self._cached_logDetJ = ((x0.size(-1) - 1) * torch.log1p(beta_h) +
                                torch.log1p(beta_h + beta * h_prime * r)).sum(-1)
        return x + beta_h * diff

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. As noted above, this implementation is incapable of
        inverting arbitrary values `y`; rather it assumes `y` is the result of a
        previously computed application of the bijector to some `x` (which was
        cached on the forward call)
        """

        raise KeyError("ConditionedRadial object expected to find key in intermediates cache but didn't")

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


@copy_docs_from(ConditionedRadial)
class Radial(ConditionedRadial, TransformModule):
    r"""
    A 'radial' bijective transform using the equation,

        :math:`\mathbf{y} = \mathbf{x} + \beta h(\alpha,r)(\mathbf{x} - \mathbf{x}_0)`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    and the learnable parameters are :math:`\alpha\in\mathbb{R}^+`,
    :math:`\beta\in\mathbb{R}`, :math:`\mathbf{x}_0\in\mathbb{R}^D`, for input
    dimension :math:`D`, :math:`r=||\mathbf{x}-\mathbf{x}_0||_2`,
    :math:`h(\alpha,r)=1/(\alpha+r)`. For this to be an invertible transformation,
    the condition :math:`\beta>-\alpha` is enforced.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = Radial(10)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse of this transform does not possess an analytical solution and is
    left unimplemented. However, the inverse is cached when the forward operation is
    called during sampling, and so samples drawn using the radial transform can be
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

        self.x0 = nn.Parameter(torch.Tensor(input_dim,))
        self.alpha_prime = nn.Parameter(torch.Tensor(1,))
        self.beta_prime = nn.Parameter(torch.Tensor(1,))
        self.input_dim = input_dim
        self.reset_parameters()

    def _params(self):
        return self.x0, self.alpha_prime, self.beta_prime

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.x0.size(0))
        self.alpha_prime.data.uniform_(-stdv, stdv)
        self.beta_prime.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)


@copy_docs_from(ConditionalTransformModule)
class ConditionalRadial(ConditionalTransformModule):
    r"""
    A conditional 'radial' bijective transform context using the equation,

        :math:`\mathbf{y} = \mathbf{x} + \beta h(\alpha,r)(\mathbf{x} - \mathbf{x}_0)`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    and :math:`\alpha\in\mathbb{R}^+`, :math:`\beta\in\mathbb{R}`,
    and :math:`\mathbf{x}_0\in\mathbb{R}^D`, are the output of a function, e.g. a NN,
    with input :math:`z\in\mathbb{R}^{M}` representing the context variable to
    condition on. The input dimension is :math:`D`,
    :math:`r=||\mathbf{x}-\mathbf{x}_0||_2`, and :math:`h(\alpha,r)=1/(\alpha+r)`.
    For this to be an invertible transformation, the condition :math:`\beta>-\alpha`
    is enforced.

    Example usage:

    >>> from pyro.nn.dense_nn import DenseNN
    >>> input_dim = 10
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [input_dim, 1, 1]
    >>> hypernet = DenseNN(context_dim, [50, 50], param_dims)
    >>> transform = ConditionalRadial(hypernet)
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size])) # doctest: +SKIP

    The inverse of this transform does not possess an analytical solution and is
    left unimplemented. However, the inverse is cached when the forward operation is
    called during sampling, and so samples drawn using the radial transform can be
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

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def _params(self, context):
        return self.nn(context)

    def condition(self, context):
        params = partial(self._params, context)
        return ConditionedRadial(params)


def radial(input_dim):
    """
    A helper function to create a :class:`~pyro.distributions.transforms.Radial`
    object for consistency with other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int

    """

    return Radial(input_dim)


def conditional_radial(input_dim, context_dim, hidden_dims=None):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.ConditionalRadial` object that takes care
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
    nn = DenseNN(context_dim, hidden_dims, param_dims=[input_dim, 1, 1])
    return ConditionalRadial(nn)
