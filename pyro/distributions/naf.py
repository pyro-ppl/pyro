from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import constraints
import torch.nn.functional as F

from pyro.distributions.util import copy_docs_from

eps = 1e-6


@copy_docs_from(TransformModule)
class DeepSigmoidalFlow(TransformModule):
    """
    An implementation of deep sigmoidal flow (DSF) Neural Autoregressive Flow (NAF), of the "IAF flavour"
    that can be used for sampling and scoring samples drawn from it (but not arbitrary ones).

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> arn = AutoRegressiveNN(10, [40], param_dims=[16]*3)
    >>> naf = DeepSigmoidalFlow(arn, hidden_units=16)
    >>> pyro.module("my_naf", naf)  # doctest: +SKIP
    >>> naf_dist = dist.TransformedDistribution(base_dist, [naf])
    >>> naf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse operation is not implemented. This would require numerical inversion, e.g., using a
    root finding method - a possibility for a future implementation.

    :param autoregressive_nn: an autoregressive neural network whose forward call returns a tuple of three
        real-valued tensors, whose last dimension is the input dimension, and whose penultimate dimension
        is equal to hidden_units.
    :type autoregressive_nn: nn.Module
    :param hidden_units: the number of hidden units to use in the NAF transformation (see Eq (8) in reference)
    :type hidden_units: int

    Reference:

    Neural Autoregressive Flows [arXiv:1804.00779]
    Chin-Wei Huang, David Krueger, Alexandre Lacoste, Aaron Courville

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, autoregressive_nn, hidden_units=16):
        super(DeepSigmoidalFlow, self).__init__(cache_size=1)
        self.arn = autoregressive_nn
        self.hidden_units = hidden_units
        self._cached_A = None
        self._cached_C = None
        self._cached_D = None
        self._cached_W_pre = None

        # Not entirely sure this is necessary, but copying from NAF paper's implementation
        self.safe_log = lambda x: torch.log(x * 1e2) - math.log(1e2)
        self.safe_logit = lambda x: self.safe_log(x) - self.safe_log(1 - x)

        # The output of safe_sigmoid is between [0.5*eps, 1-0.5*eps], which stops logit from returning +-Inf (overflow)
        self.sigmoid = nn.Sigmoid()
        self.safe_sigmoid = lambda x: self.sigmoid(x) * (1 - eps) + 0.5 * eps
        self.logsoftmax = nn.LogSoftmax(dim=-2)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        # A, W, b ~ batch_shape x hidden_units x event_shape
        A, W_pre, b = self.arn(x)

        # Divide the autoregressive output into the component activations
        A = F.softplus(A)
        C = A * x.unsqueeze(-2) + b
        W = F.softmax(W_pre, dim=-2)
        D = (W * self.safe_sigmoid(C)).sum(dim=-2)

        # The use of a special sigmoid here is so that logit doesn't overflow
        # NOTE: Element-wise multiplication by W then summing over second-last dim is equivalent to
        # dot-product over cols (or rows?) and cols (or rows?) of sigmoid term
        # The unsqueeze on `x` is so that A * x.unsqueeze(-2) broadcasts correctly over hidden_units dimension
        y = self.safe_logit(D)
        self._cached_W_pre = W_pre
        self._cached_A = A
        self._cached_C = C
        self._cached_D = D
        return y

    # This method returns log(abs(det(dy/dx)), which is equal to -log(abs(det(dx/dy))
    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        # Need W_pre, A, C, D
        W_pre = self._cached_W_pre
        A = self._cached_A
        C = self._cached_C
        D = self._cached_D

        # See C.1 of Huang Et Al. for a derivation of this
        # NOTE: Since safe_logit is mathematically the same as logit, this line doesn't need any modification
        log_dydD = -torch.log(D + 1e-8) - torch.log(1 - D)

        # NOTE: However, safe_sigmoid differs from torch.sigmoid, which we need to take into account in the derivative!
        # Hence the log(1 - eps) term
        log_dDdx = torch.logsumexp(self.logsoftmax(W_pre) + F.logsigmoid(C) + F.logsigmoid(-C) +
                                   torch.log1p(torch.tensor(-eps)) + torch.log(A), dim=-2)
        log_det = log_dydD + log_dDdx

        return log_det.sum(-1)
