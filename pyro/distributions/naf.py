from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform
from torch.distributions import constraints
import torch.nn.functional as F

from pyro.distributions.util import copy_docs_from

eps = 1e-6


@copy_docs_from(Transform)
class NeuralAutoregressiveSampling(Transform):
    """
    An implementation of deep sigmoidal flow (DSF) Neural Autoregressive Flow (NAF), of the "IAF flavour"
    that can be used for sampling and scoring samples drawn from it (but not arbitrary ones).

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> arn = AutoRegressiveNN(10, [40], param_dims=[16]*3)
    >>> naf = NeuralAutoregressiveSampling(arn, hidden_units=16)
    >>> naf_module = pyro.module("my_naf", naf.module)
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

    codomain = constraints.real

    def __init__(self, autoregressive_nn, hidden_units=16):
        super(NeuralAutoregressiveSampling, self).__init__()
        self.module = nn.Module()
        self.module.arn = autoregressive_nn
        self.hidden_units = hidden_units
        self._intermediates_cache = {}
        self.add_inverse_to_cache = True
        self.logsoftmax = nn.LogSoftmax(dim=-2)
        self.logsigmoid = nn.LogSigmoid()
        self.safe_log = lambda x: torch.log(x * 1e2) - math.log(1e2)
        self.safe_logit = lambda x: self.safe_log(x) - self.safe_log(1 - x)
        self.sigmoid = nn.Sigmoid()
        self.safe_sigmoid = lambda x: self.sigmoid(x) * (1 - eps) + 0.5 * eps

    @property
    def arn(self):
        """
        :rtype: pyro.nn.AutoRegressiveNN

        Return the AutoRegressiveNN associated with the InverseAutoregressiveFlow
        """
        return self.module.arn

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        A, W, b = self.module.arn(x)

        # Divide the autoregressive output into the component activations
        A = F.softplus(A)
        W = F.softmax(W, dim=-2)

        # The use of a special sigmoid here is so that logit doesn't overflow
        y = self.safe_logit(torch.sum(W * self.safe_sigmoid(A * x.unsqueeze(-2) + b), dim=-2))
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
            raise KeyError("NeuralAutoregressiveSampling expected to find "
                           "key in intermediates cache but didn't")

    def _add_intermediate_to_cache(self, intermediate, y, name):
        """
        Internal function used to cache intermediate results computed during the forward call
        """
        assert((y, name) not in self._intermediates_cache),\
            "key collision in _add_intermediate_to_cache"
        self._intermediates_cache[(y, name)] = intermediate

    # This method returns log(abs(det(dy/dx)), which is equal to -log(abs(det(dx/dy))
    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        A, W_pre, b = self.module.arn(x)

        A = F.softplus(A)
        W = F.softmax(W_pre, dim=-2)

        C = A * x.unsqueeze(-2) + b
        D = (W * self.safe_sigmoid(C)).sum(dim=-2)
        D_clip = D.clamp(min=eps / 2, max=1. - eps / 2)

        # See C.1 of Huang Et Al. for a derivation of this
        log_dydD = -torch.log(D_clip) - torch.log(1 - D_clip)
        log_dDdx = torch.logsumexp(self.logsoftmax(W_pre) + self.logsigmoid(C) + self.logsigmoid(-C)
                                   + torch.log(A), dim=-2)
        log_det = log_dydD + log_dDdx

        return log_det


@copy_docs_from(Transform)
class NeuralAutoregressiveScoring(Transform):
    """
    An implementation of deep sigmoidal flow (DSF) Neural Autoregressive Flow (NAF), of the "MAF flavour"
    that can be used for density estimation but not sampling.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> target_dist = dist.Normal(5*torch.ones(10), 0.25*torch.ones(10))
    >>> arn = AutoRegressiveNN(10, [40], param_dims=[16]*3)
    >>> naf = NeuralAutoregressiveScoring(arn, hidden_units=16)
    >>> naf_dist = dist.TransformedDistribution(base_dist, [naf])
    >>> naf_dist.log_prob(target_dist.sample())  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The forward operation is not implemented. This would require numerical inversion, e.g., using a
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

    codomain = constraints.real

    def __init__(self, autoregressive_nn, hidden_units=16):
        super(NeuralAutoregressiveScoring, self).__init__()
        self.module = nn.Module()
        self.module.arn = autoregressive_nn
        self.hidden_units = hidden_units
        self._intermediates_cache = {}
        self.add_inverse_to_cache = True
        self.logsoftmax = nn.LogSoftmax(dim=-2)
        self.logsigmoid = nn.LogSigmoid()
        self.safe_log = lambda x: torch.log(x * 1e2) - math.log(1e2)
        self.safe_logit = lambda x: self.safe_log(x) - self.safe_log(1 - x)
        self.sigmoid = nn.Sigmoid()
        self.safe_sigmoid = lambda x: self.sigmoid(x) * (1 - eps) + 0.5 * eps

    @property
    def arn(self):
        """
        :rtype: pyro.nn.AutoRegressiveNN

        Return the AutoRegressiveNN associated with the InverseAutoregressiveFlow
        """
        return self.module.arn

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """

        raise NotImplementedError("Forward operation is not implemented for NeuralAutoregressiveScoring")

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """

        A, W, b = self.module.arn(y)

        # Divide the autoregressive output into the component activations
        A = F.softplus(A)
        W = F.softmax(W, dim=-2)

        # The use of a special sigmoid here is so that logit doesn't overflow
        x = self.safe_logit(torch.sum(W * self.safe_sigmoid(A * y.unsqueeze(-2) + b), dim=-2))
        return x

    # This method returns log(abs(det(dy/dx)), which is equal to -log(abs(det(dx/dy))
    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        A, W_pre, b = self.module.arn(y)
        A = F.softplus(A)
        W = F.softmax(W_pre, dim=-2)

        C = A * y.unsqueeze(-2) + b
        D = (W * self.safe_sigmoid(C)).sum(dim=-2)
        D_clip = D.clamp(min=eps / 2, max=1. - eps / 2)

        # See C.1 of Huang Et Al. for a derivation of this
        log_dxdD = -torch.log(D_clip) - torch.log(1 - D_clip)
        log_dDdy = torch.logsumexp(self.logsoftmax(W_pre) + self.logsigmoid(C)
                                   + self.logsigmoid(-C) + torch.log(A), dim=-2)
        log_det = log_dxdD + log_dDdy

        return -log_det
