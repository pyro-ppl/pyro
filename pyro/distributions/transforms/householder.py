# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TransformModule)
class Householder(TransformModule):
    """
    Represents multiple applications of the Householder bijective transformation. A
    single Householder transformation takes the form,

        :math:`\\mathbf{y} = (I - 2*\\frac{\\mathbf{u}\\mathbf{u}^T}{||\\mathbf{u}||^2})\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    and the learnable parameters are :math:`\\mathbf{u}\\in\\mathbb{R}^D` for input
    dimension :math:`D`.

    The transformation represents the reflection of :math:`\\mathbf{x}` through the
    plane passing through the origin with normal :math:`\\mathbf{u}`.

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

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1
    volume_preserving = True

    def __init__(self, input_dim, count_transforms=1):
        super().__init__(cache_size=1)

        self.input_dim = input_dim
        if count_transforms < 1:
            raise ValueError('Number of Householder transforms, {}, is less than 1!'.format(count_transforms))
        elif count_transforms > input_dim:
            warnings.warn(
                "Number of Householder transforms, {}, is greater than input dimension {}, which is an \
over-parametrization!".format(count_transforms, input_dim))
        self.count_transforms = count_transforms
        self.u_unnormed = nn.Parameter(torch.Tensor(count_transforms, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u_unnormed.size(-1))
        self.u_unnormed.data.uniform_(-stdv, stdv)

    # Construct normalized vectors for Householder transform
    def u(self):
        norm = torch.norm(self.u_unnormed, p=2, dim=-1, keepdim=True)
        return torch.div(self.u_unnormed, norm)

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
        for idx in range(self.count_transforms):
            projection = (u[idx] * y).sum(dim=-1, keepdim=True) * u[idx]
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
        for jdx in reversed(range(self.count_transforms)):
            # NOTE: Need to apply transforms in reverse order as forward operation!
            projection = (u[jdx] * x).sum(dim=-1, keepdim=True) * u[jdx]
            x = x - 2. * projection
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian. Householder flow
        is measure preserving, so :math:`\\log(|detJ|) = 0`
        """

        return torch.zeros(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device)


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
