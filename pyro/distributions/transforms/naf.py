from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import constraints
import torch.nn.functional as F

from pyro.distributions.util import copy_docs_from

eps = 1e-8


class ELUMixin(object):
    @staticmethod
    def f(x):
        """
        Implements the nonlinearity of NAF, in this case ELU
        """
        return F.elu(x)

    @staticmethod
    def f_inv(x):
        """
        Implements the inverse of ELU
        """
        return torch.max(x, torch.zeros_like(x)) + torch.min(torch.log1p(x + eps), torch.zeros_like(x))

    @staticmethod
    def log_df_dx(x):
        """
        Implements the log derivative of NAF nonlinearity
        """
        return -F.relu(-x)

    @staticmethod
    def log_df_inv_dx(x):
        """
        Implements the log derivative of inverse NAF nonlinearity
        """
        return F.relu(-torch.log1p(x + eps))


class LeakyReLUMixin(object):
    @staticmethod
    def f(x):
        """
        Implements the nonlinearity of NAF, in this case leaky ReLU
        """
        return F.leaky_relu(x)

    @staticmethod
    def f_inv(x):
        """
        Implements the inverse of leaky ReLU
        """
        # slope for negative part is inverse of slope for positive part in f(x)
        return F.leaky_relu(x, negative_slope=100.0)

    @staticmethod
    def log_df_dx(x):
        """
        Implements the log derivative of NAF nonlinearity
        """
        return torch.where(x >= 0., torch.zeros_like(x), torch.ones_like(x) * math.log(0.01))

    @staticmethod
    def log_df_inv_dx(x):
        """
        Implements the log derivative of inverse NAF nonlinearity
        """
        return torch.where(x >= 0., torch.zeros_like(x), torch.ones_like(x) * math.log(100.0))


class SigmoidalMixin(object):
    @staticmethod
    def f(x):
        """
        Implements the nonlinearity of NAF, in this case sigmoid with scaled output
        """
        return torch.sigmoid(x) * (1. - eps) + 0.5 * eps

    @staticmethod
    def f_inv(x):
        """
        Implements the inverse scaled sigmoid nonlinearity
        """
        y = (x - 0.5 * eps) / (1. - eps)
        return torch.log(y) - torch.log1p(-y)

    @staticmethod
    def log_df_dx(x):
        """
        Implements the log derivative of scaled sigmoid nonlinearity
        """
        return F.logsigmoid(x) + F.logsigmoid(-x) + torch.log1p(torch.tensor(-eps))

    @staticmethod
    def log_df_inv_dx(x):
        """
        Implements the log derivative of inverse scaled sigmoid nonlinearity
        """
        y = (x - 0.5 * eps) / (1. - eps)
        return -torch.log(y + eps) - torch.log(1. - y) - math.log(1. - eps)


class TanhMixin(object):
    @staticmethod
    def f(x):
        """
        The nonlinearity to apply after each masked block linear layer
        """
        return torch.tanh(x)

    @staticmethod
    def log_df_dx(x):
        return - 2. * (x - math.log(2.) + F.softplus(- 2. * x))


@copy_docs_from(TransformModule)
class DeepNAFFlow(TransformModule):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1
    autoregressive = True

    def __init__(self, autoregressive_nn, hidden_units=16):
        super(DeepNAFFlow, self).__init__(cache_size=1)
        self.arn = autoregressive_nn
        self.hidden_units = hidden_units
        self.logsoftmax = nn.LogSoftmax(dim=-2)
        self._cached_A = None
        self._cached_W_pre = None
        self._cached_C = None
        self._cached_D = None

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
        D = (W * self.f(C)).sum(dim=-2)
        y = self.f_inv(D)

        self._cached_A = A
        self._cached_W_pre = W_pre
        self._cached_C = C
        self._cached_D = D

        return y

    # This method returns log(abs(det(dy/dx)), which is equal to -log(abs(det(dx/dy))
    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """

        A = self._cached_A
        W_pre = self._cached_W_pre
        C = self._cached_C
        D = self._cached_D

        log_dydD = self.log_df_inv_dx(D)
        log_dDdx = torch.logsumexp(torch.log(A + eps) + self.logsoftmax(W_pre) + self.log_df_dx(C), dim=-2)
        log_det = log_dydD + log_dDdx
        return log_det.sum(-1)

    @staticmethod
    def f(x):
        """
        Implements the nonlinearity of NAF
        """
        raise NotImplementedError

    @staticmethod
    def f_inv(x):
        """
        Implements the inverse of the NAF nonlinearity
        """
        raise NotImplementedError

    @staticmethod
    def log_df_dx(x):
        """
        Implements the log derivative of NAF nonlinearity
        """
        raise NotImplementedError

    @staticmethod
    def log_df_inv_dx(x):
        """
        Implements the log derivative of inverse NAF nonlinearity
        """
        raise NotImplementedError


@copy_docs_from(TransformModule)
class DeepELUFlow(ELUMixin, DeepNAFFlow):
    """
    An implementation of deep ELU flow (DSF) Neural Autoregressive Flow (NAF), of the "IAF flavour"
    that can be used for sampling and scoring samples drawn from it (but not arbitrary ones). This
    flow is suggested in Huang et al., 2018, section 3.3, but left for future experiments.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> arn = AutoRegressiveNN(10, [40], param_dims=[16]*3)
    >>> naf = DeepELUFlow(arn, hidden_units=16)
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
    pass


@copy_docs_from(TransformModule)
class DeepSigmoidalFlow(SigmoidalMixin, DeepNAFFlow):
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
    pass


@copy_docs_from(TransformModule)
class DeepLeakyReLUFlow(LeakyReLUMixin, DeepNAFFlow):
    """
    An implementation of deep leaky ReLU flow (DSF) Neural Autoregressive Flow (NAF), of the "IAF flavour"
    that can be used for sampling and scoring samples drawn from it (but not arbitrary ones). This
    flow is suggested in Huang et al., 2018, section 3.3, but left for future experiments.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> arn = AutoRegressiveNN(10, [40], param_dims=[16]*3)
    >>> naf = DeepLeakyReLUFlow(arn, hidden_units=16)
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
    pass
