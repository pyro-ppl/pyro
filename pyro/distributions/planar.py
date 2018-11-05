from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform
from torch.distributions import constraints
import torch.nn.functional as F

from pyro.distributions.util import copy_docs_from

# This helper function clamps gradients but still passes through the gradient in clamped regions
# NOTE: Not sure how necessary this is, but I was copying the design of the TensorFlow implementation


def clamp_preserve_gradients(x, min, max):
    return x + (x.clamp(min, max) - x).detach()


@copy_docs_from(Transform)
class PlanarFlow(Transform):
    """
    A 'planar' normalizing flow that uses the transformation

        :math:`\\mathbf{y} = \\mathbf{x} + \\mathbf{u}\\tanh(\\mathbf{w}^T\\mathbf{z}+b)`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, and the learnable parameters
    are :math:`b\\in\\mathbb{R}`, :math:`\\mathbf{u}\\in\\mathbb{R}^D`, :math:`\\mathbf{w}\\in\\mathbb{R}^D` for input
    dimension :math:`D`. For this to be an invertible transformation, the condition 
    :math:`\\mathbf{w}^T\\mathbf{u}>-1` is enforced.

    Together with `TransformedDistribution` this provides a way to create richer variational approximations.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> plf = PlanarFlow(10)
    >>> plf_module = pyro.module("my_plf", plf.module)
    >>> plf_dist = dist.TransformedDistribution(base_dist, [plf])
    >>> plf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse of this transform does not possess an analytical solution and is left unimplemented. However,
    the inverse is cached when the forward operation is called during sampling, and so samples drawn using
    planar flow can be scored.

    References:

    Variational Inference with Normalizing Flows [arXiv:1505.05770]
    Danilo Jimenez Rezende, Shakir Mohamed

    """

    codomain = constraints.real

    def __init__(self, input_dim):
        super(PlanarFlow, self).__init__()
        self.input_dim = input_dim
        self.module = nn.Module()
        self.module.lin = nn.Linear(input_dim, 1)
        self.module.u = nn.Parameter(torch.Tensor(input_dim))
        self.reset_parameters()
        self._intermediates_cache = {}
        self.add_inverse_to_cache = True

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.module.u.size(0))
        self.module.lin.weight.data.uniform_(-stdv, stdv)

    def u_hat(self):
      u = self.module.u

      # TODO: Reshape W?
      w = self.module.lin.weight.squeeze(0)

      alpha = torch.dot(u, w)
      a_prime = -1 + F.softplus(alpha)

      return u + (a_prime - alpha) * w.div(w.norm())
    
    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """

        y = x + self.u_hat() * torch.tanh(self.module.lin(x))

        self._add_intermediate_to_cache(x, y, 'x')
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """
        if (y, 'x') in self._intermediates_cache:
            x = self._intermediates_cache.pop((y, 'x'))
            return x
        else:
            raise KeyError("PlanarFlow expected to find "
                           "key in intermediates cache but didn't")

    def _add_intermediate_to_cache(self, intermediate, y, name):
        """
        Internal function used to cache intermediate results computed during the forward call
        """
        assert((y, name) not in self._intermediates_cache),\
            "key collision in _add_intermediate_to_cache"
        self._intermediates_cache[(y, name)] = intermediate

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        psi_z = (1 - torch.tanh(self.module.lin(x)).pow(2))*self.module.lin.weight

        # TODO: Check that dimensions of W broadcast properly!
        #print('W', self.module.lin.weight.size(), 'psi_z', psi_z.size(), 'u', self.module.u.size())
        #raise Exception()

        # TODO: Continue from here, 5/11/2018!
        # *** Need to take account of fact that psi_z has a batch dimension
        #return torch.abs(1 + torch.dot(self.u_hat(), psi_z))
        return torch.abs(1 + psi_z * self.u_hat())
