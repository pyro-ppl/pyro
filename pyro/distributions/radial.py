from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from torch.distributions import constraints
import torch.nn.functional as F

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TransformModule)
class RadialFlow(TransformModule):
    """
    A 'radial' normalizing flow that uses the transformation

        :math:`\\mathbf{y} = \\mathbf{x} + \\mathbf{u}\\tanh(\\mathbf{w}^T\\mathbf{z}+b)`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, and the learnable parameters
    are :math:`b\\in\\mathbb{R}`, :math:`\\mathbf{u}\\in\\mathbb{R}^D`, :math:`\\mathbf{w}\\in\\mathbb{R}^D` for input
    dimension :math:`D`. For this to be an invertible transformation, the condition
    :math:`\\mathbf{w}^T\\mathbf{u}>-1` is enforced.

    Together with `TransformedDistribution` this provides a way to create richer variational approximations.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> plf = PlanarFlow(10)
    >>> pyro.module("my_plf", plf)  # doctest: +SKIP
    >>> plf_dist = dist.TransformedDistribution(base_dist, [plf])
    >>> plf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse of this transform does not possess an analytical solution and is left unimplemented. However,
    the inverse is cached when the forward operation is called during sampling, and so samples drawn using
    planar flow can be scored.

    :param input_dim: the dimension of the input (and output) variable.
    :type autoregressive_nn: int

    References:

    Variational Inference with Normalizing Flows [arXiv:1505.05770]
    Danilo Jimenez Rezende, Shakir Mohamed

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, input_dim):
        super(RadialFlow, self).__init__(cache_size=1)

        self.input_dim = input_dim
        #self.lin = nn.Linear(input_dim, 1)
        self.x0 = nn.Parameter(torch.Tensor(input_dim))
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.x0.size(0))
        self.alpha.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)

    # This method ensures that beta > -alpha, required for invertibility
    def beta_prime(self):
        return -self.alpha + F.softplus(self.beta)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        r = (x - self.x0).norm(dim=-1, keepdim=True)
        h = (self.alpha + r).reciprocal()
        y = x + self.beta_prime() * h * (x - self.x0)
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """

        raise KeyError("RadialFlow expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        r = (x - self.x0).norm(dim=-1, keepdim=True)
        h = (self.alpha + r).reciprocal()
        h_prime = -(self.alpha + r).pow(-2)
        beta_prime = self.beta_prime()

        first_term = (self.input_dim - 1) * torch.log1p(beta_prime*h) 
        second_term = torch.log1p(beta_prime*h + h_prime*r)

        return (first_term + second_term).sum(-1)
