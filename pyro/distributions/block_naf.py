# This implementation uses some code from https://github.com/nicola-decao/BNAF under the MIT license.
from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import constraints
import torch.nn.functional as F

from pyro.distributions.util import copy_docs_from

eps = 1e-8


def log_matrix_product(A, B):
    """
    Computes the matrix products of two matrices in log-space, returning the result in log-space.
    This is useful for calculating the vector chain rule for Jacobian terms.
    """
    return torch.logsumexp(A.unsqueeze(-1) + B.unsqueeze(-3), dim=-2)


@copy_docs_from(TransformModule)
class BlockNAFFlow(TransformModule):
    """
    An implementation of Block Neural Autoregressive Flow (block-NAF) (De Cao et al., 2019). Block-NAF uses a
    similar transformation to deep dense NAF, building the autoregressive NN into the structure of the flow,
    in a sense.

    Together with `TransformedDistribution` this provides a way to create richer variational approximations.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> naf = BlockNAFFlow(input_dim=10)
    >>> pyro.module("my_naf", naf)  # doctest: +SKIP
    >>> naf_dist = dist.TransformedDistribution(base_dist, [naf])
    >>> naf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse operation is not implemented. This would require numerical inversion, e.g., using a
    root finding method - a possibility for a future implementation.

    :param input_dim: The dimensionality of the input and output variables.
    :type input_dim: int
    :param hidden_factors: Hidden layer i has hidden_factors[i] hidden units per input dimension. This
        corresponds to both :math:`a` and :math:`b` in De Cao et al. (2019). The elements of hidden_factors
        must be integers.
    :type hidden_factors: list
    :param residual: Type of residual connections to use. Choices are "None", "normal" for
        :math:`\\mathbf{y}+f(\\mathbf{y})`, and "gated" for :math:`\\alpha\\mathbf{y} + (1 - \\alpha\\mathbf{y})`
        for learnable parameter :math:`\\alpha`.
    :type residual: string

    References:

    Block Neural Autoregressive Flow [arXiv:1904.04676]
    Nicola De Cao, Ivan Titov, Wilker Aziz
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1
    autoregressive = True

    def __init__(self, input_dim, hidden_factors=[8, 8], residual=None):
        super(BlockNAFFlow, self).__init__(cache_size=1)

        if any([h < 1 for h in hidden_factors]):
            raise ValueError('Hidden factors, {}, must all be >= 1'.format(hidden_factors))

        if residual not in [None, 'normal', 'gated']:
            raise ValueError('Invalid value {} for keyword argument "residual"'.format(residual))

        # Initialize modules for each layer in flow
        self.residual = residual
        self.input_dim = input_dim
        self.layers = nn.ModuleList([MaskedBlockLinear(input_dim, input_dim * hidden_factors[0], input_dim)])
        for idx in range(1, len(hidden_factors)):
            self.layers.append(MaskedBlockLinear(
                input_dim * hidden_factors[idx - 1], input_dim * hidden_factors[idx], input_dim))
        self.layers.append(MaskedBlockLinear(input_dim * hidden_factors[-1], input_dim, input_dim))
        self._cached_logDetJ = None

        if residual == 'gated':
            self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))

    def f(self, x):
        """
        The nonlinearity to apply after each masked block linear layer
        """
        return torch.tanh(x)

    def log_df_dx(self, x):
        return - 2. * (x - math.log(2.) + F.softplus(- 2. * x))

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        y = x
        for idx in range(len(self.layers)):
            pre_activation, dy_dx = self.layers[idx](y.unsqueeze(-1))
            J_act = self.log_df_dx(pre_activation).view(*(list(x.size()) + [-1, 1]))

            if idx == 0:
                logDetJ = dy_dx + J_act
            else:
                logDetJ = log_matrix_product(dy_dx, logDetJ) + J_act

            y = self.f(pre_activation)

        self._cached_logDetJ = logDetJ.squeeze(-1).squeeze(-1)

        if self.residual == 'normal':
            y += x
            self._cached_logDetJ = F.softplus(self._cached_logDetJ)
        elif self.residual == 'gated':
            y = self.gate.sigmoid() * x + (1. - self.gate.sigmoid()) * y
            term1 = torch.log(self.gate.sigmoid() + eps)
            log1p_gate = torch.log1p(eps - self.gate.sigmoid())
            log_gate = torch.log(self.gate.sigmoid() + eps)
            term2 = F.softplus(log1p_gate - log_gate + self._cached_logDetJ)
            self._cached_logDetJ = term1 + term2

        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """

        raise KeyError("BlockNAFFlow expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        return self._cached_logDetJ.sum(-1)


class MaskedBlockLinear(torch.nn.Module):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """

    def __init__(self, in_features, out_features, dim, bias=True):
        super(MaskedBlockLinear, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim

        weight = torch.zeros(out_features, in_features)

        # Fill in non-zero entries of block weight matrix, going from top
        # to bottom.
        for i in range(dim):
            weight[i * out_features // dim:(i + 1) * out_features // dim,
                   0:(i + 1) * in_features // dim] = torch.nn.init.xavier_uniform_(
                torch.Tensor(out_features // dim, (i + 1) * in_features // dim))

        self._weight = torch.nn.Parameter(weight)
        self._diag_weight = torch.nn.Parameter(torch.nn.init.uniform_(torch.Tensor(out_features, 1)).log())

        self.bias = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.Tensor(out_features),
                                   -1 / math.sqrt(out_features),
                                   1 / math.sqrt(out_features))) if bias else 0

        # Diagonal block mask
        mask_d = torch.eye(dim).unsqueeze(-1).repeat(1, out_features // dim,
                                                     in_features // dim).view(out_features, in_features)
        self.register_buffer('mask_d', mask_d)

        # Off-diagonal block mask for lower triangular weight matrix
        mask_o = torch.tril(torch.ones(dim, dim), diagonal=-1).unsqueeze(-1)
        mask_o = mask_o.repeat(1, out_features // dim, in_features // dim).view(out_features, in_features)
        self.register_buffer('mask_o', mask_o)

    def get_weights(self):
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """

        # Form block weight matrix, making sure it's positive on diagonal!
        w = torch.exp(self._weight) * self.mask_d + self._weight * self.mask_o

        # NOTE: Commented out weight normalization for now!
        # Sum is taken over columns, i.e. one norm per row
        w_squared_norm = (w ** 2).sum(-1, keepdim=True)

        # Effect of multiplication and division is that each row is normalized and rescaled
        w = self._diag_weight.exp() * w / (w_squared_norm.sqrt() + eps)

        # Taking the effect of weight normalization into account in calculating the log-gradient is straightforward!
        # Instead of differenting, e.g. d(W_1x)/dx, we have d(g_1W_1/(W_1^TW_1)^0.5x)/dx, roughly speaking, and taking
        # the log gives the right hand side below:
        wpl = self._diag_weight + self._weight - 0.5 * torch.log(w_squared_norm + eps)

        return w, wpl[self.mask_d.byte()].view(self.dim, self.out_features // self.dim, self.in_features // self.dim)

    def forward(self, x):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal block of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """

        w, wpl = self.get_weights()
        return (torch.matmul(w, x) + self.bias.unsqueeze(-1)).squeeze(-1), wpl
