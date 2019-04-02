from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TransformModule)
class HouseholderFlow(TransformModule):
    """
    A single transformation of Householder flow,

        :math:`\\mathbf{y} = (I - 2*\\frac{\\mathbf{u}\\mathbf{u}^T}{||\\mathbf{u}||^2})\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, and the learnable parameters
    are :math:`\\mathbf{u}\\in\\mathbb{R}^D` for input dimension :math:`D`.

    The transformation represents the reflection of :math:`\\mathbf{x}` through the plane passing through the
    origin with normal :math:`\\mathbf{u}`. Together with `TransformedDistribution` this provides a way to
    create richer variational approximations.

    :math:`D` applications of this transformation are able to transform standard i.i.d. standard Gaussian noise
    into a Gaussian variable with an arbitrary covariance matrix. With :math:`K<D` transformations, one is able
    to approximate a full-rank covariance Gaussian distribution.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> flows = [HouseholderFlow(10) for _ in range(3)]
    >>> [pyro.module("my_flow", p) for f in flows] # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, flows)
    >>> flow_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int

    References:

    Improving Variational Auto-Encoders using Householder Flow, [arXiv:1611.09630]
    Tomczak, J. M., & Welling, M.

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, input_dim):
        super(HouseholderFlow, self).__init__(cache_size=1)

        self.input_dim = input_dim
        self.u = nn.Parameter(torch.Tensor(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u.size(0))
        self.u.data.uniform_(-stdv, stdv)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """

        squared_norm = self.u.pow(2).sum(-1)
        projection = (self.u * x).sum(dim=-1, keepdim=True) * self.u / squared_norm
        y = x - 2. * projection
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """

        # The Householder transformation, H, is "involutory," i.e. H^2 = I
        # If you reflect a point around a plane, then the same operation will reflect it back
        return self._call(y)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian. Householder flow is measure preserving,
        so log(|detJ|) = 0
        """

        return torch.zeros(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device)
