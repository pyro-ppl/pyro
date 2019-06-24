from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.transforms.householder import HouseholderFlow
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TransformModule)
class SylvesterFlow(HouseholderFlow):
    """
    An implementation of Sylvester flow of the Householder variety (Van den Berg Et Al., 2018),

        :math:`\\mathbf{y} = \\mathbf{x} + QR\\tanh(SQ^T\\mathbf{x}+\\mathbf{b})`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, :math:`R,S\\sim D\\times D`
    are upper triangular matrices for input dimension :math:`D`, :math:`Q\\sim D\\times D` is an orthogonal
    matrix, and :math:`\\mathbf{b}\\sim D` is learnable bias term.

    Sylvester flow is a generalization of Planar flow. In the Householder type of Sylvester flow, the
    orthogonality of :math:`Q` is enforced by representing it as the product of Householder transformations

    Together with `TransformedDistribution` it provides a way to create richer variational approximations.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> hsf = SylvesterFlow(10, count_transforms=4)
    >>> pyro.module("my_hsf", hsf)  # doctest: +SKIP
    >>> hsf_dist = dist.TransformedDistribution(base_dist, [hsf])
    >>> hsf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse of this transform does not possess an analytical solution and is left unimplemented. However,
    the inverse is cached when the forward operation is called during sampling, and so samples drawn using
    Sylvester flow can be scored.

    References:

    Rianne van den Berg, Leonard Hasenclever, Jakub M. Tomczak, Max Welling. Sylvester Normalizing Flows for
    Variational Inference. In proceedings of The 34th Conference on Uncertainty in Artificial Intelligence
    (UAI 2018).

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, input_dim, count_transforms=1):
        super(SylvesterFlow, self).__init__(input_dim, count_transforms)

        # Create parameters for Sylvester transform
        self.R_dense = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.S_dense = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.R_diag = nn.Parameter(torch.Tensor(input_dim))
        self.S_diag = nn.Parameter(torch.Tensor(input_dim))
        self.b = nn.Parameter(torch.Tensor(input_dim))

        # Register masks and indices
        triangular_mask = torch.triu(torch.ones(input_dim, input_dim), diagonal=1)
        self.register_buffer('triangular_mask', triangular_mask)

        self._cached_logDetJ = None
        self.tanh = nn.Tanh()
        self.reset_parameters2()

    # Derivative of hyperbolic tan
    def dtanh_dx(self, x):
        return 1. - self.tanh(x).pow(2)

    # Construct upper diagonal R matrix
    def R(self):
        return self.R_dense * self.triangular_mask + torch.diag(self.tanh(self.R_diag))

    # Construct upper diagonal S matrix
    def S(self):
        return self.S_dense * self.triangular_mask + torch.diag(self.tanh(self.S_diag))

    # Construct orthonomal matrix using Householder flow
    def Q(self, x):
        u = self.u()
        partial_Q = torch.eye(self.input_dim, dtype=x.dtype, layout=x.layout,
                              device=x.device) - 2. * torch.ger(u[0], u[0])

        for idx in range(1, self.count_transforms):
            partial_Q = torch.matmul(partial_Q, torch.eye(self.input_dim) - 2. * torch.ger(u[idx], u[idx]))

        return partial_Q

    # Self.u_unnormed is initialized in parent class
    def reset_parameters2(self):
        for v in [self.b, self.R_diag, self.S_diag, self.R_dense, self.S_dense]:
            v.data.uniform_(-0.01, 0.01)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        Q = self.Q(x)
        R = self.R()
        S = self.S()

        A = torch.matmul(Q, R)
        B = torch.matmul(S, Q.t())

        preactivation = torch.matmul(x, B) + self.b
        y = x + torch.matmul(self.tanh(preactivation), A)

        self._cached_logDetJ = torch.log1p(self.dtanh_dx(preactivation) * R.diagonal() * S.diagonal() + 1e-8).sum(-1)
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """

        raise KeyError("SylvesterFlow expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """

        return self._cached_logDetJ
