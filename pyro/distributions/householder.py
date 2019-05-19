from __future__ import absolute_import, division, print_function

import math
import warnings

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TransformModule)
class HouseholderFlow(TransformModule):
    """
    A flow formed from multiple applications of the Householder transformation. A single Householder transformation
    takes the form,

        :math:`\\mathbf{y} = (I - 2*\\frac{\\mathbf{u}\\mathbf{u}^T}{||\\mathbf{u}||^2})\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, and the learnable parameters
    are :math:`\\mathbf{u}\\in\\mathbb{R}^D` for input dimension :math:`D`.

    The transformation represents the reflection of :math:`\\mathbf{x}` through the plane passing through the
    origin with normal :math:`\\mathbf{u}`.

    :math:`D` applications of this transformation are able to transform standard i.i.d. standard Gaussian noise
    into a Gaussian variable with an arbitrary covariance matrix. With :math:`K<D` transformations, one is able
    to approximate a full-rank Gaussian distribution using a linear transformation of rank :math:`K`.

    Together with `TransformedDistribution` this provides a way to create richer variational approximations.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> flow = HouseholderFlow(10, count_transforms=5)
    >>> pyro.module("my_flow", p) # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, flow)
    >>> flow_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int
    :param count_transforms: number of applications of Householder transformation to apply.
    :type count_transforms: int

    References:

    Improving Variational Auto-Encoders using Householder Flow, [arXiv:1611.09630]
    Tomczak, J. M., & Welling, M.

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, input_dim, count_transforms=1):
        super(HouseholderFlow, self).__init__(cache_size=1)

        self.input_dim = input_dim
        assert count_transforms > 0
        if count_transforms > input_dim:
            warnings.warn(
                "Number of Householder transforms, {}, is greater than input dimension {}, which is an \
over-parametrization!".format(count_transforms, input_dim))
        self.count_transforms = count_transforms
        self.u = nn.Parameter(torch.Tensor(count_transforms, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u.size(-1))
        self.u.data.uniform_(-stdv, stdv)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """

        y = x
        squared_norm = self.u.pow(2).sum(-1)
        for idx in range(self.count_transforms):
            projection = (self.u[idx] * y).sum(dim=-1, keepdim=True) * self.u[idx] / squared_norm[idx]
            y = y - 2. * projection
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. The Householder transformation, H, is "involutory," i.e. H^2 = I. If you reflect a
        point around a plane, then the same operation will reflect it back
        """

        return self._call(y)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian. Householder flow is measure preserving,
        so :math:`\\log(|detJ|) = 0`
        """

        return torch.zeros(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device)
