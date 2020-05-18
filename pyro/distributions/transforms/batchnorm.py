# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TransformModule)
class BatchNorm(TransformModule):
    r"""
    A type of batch normalization that can be used to stabilize training in
    normalizing flows. The inverse operation is defined as

        :math:`x = (y - \hat{\mu}) \oslash \sqrt{\hat{\sigma^2}} \otimes \gamma + \beta`

    that is, the standard batch norm equation, where :math:`x` is the input,
    :math:`y` is the output, :math:`\gamma,\beta` are learnable parameters, and
    :math:`\hat{\mu}`/:math:`\hat{\sigma^2}` are smoothed running averages of
    the sample mean and variance, respectively. The constraint :math:`\gamma>0` is
    enforced to ease calculation of the log-det-Jacobian term.

    This is an element-wise transform, and when applied to a vector, learns two
    parameters (:math:`\gamma,\beta`) for each dimension of the input.

    When the module is set to training mode, the moving averages of the sample mean
    and variance are updated every time the inverse operator is called, e.g., when a
    normalizing flow scores a minibatch with the `log_prob` method.

    Also, when the module is set to training mode, the sample mean and variance on
    the current minibatch are used in place of the smoothed averages,
    :math:`\hat{\mu}` and :math:`\hat{\sigma^2}`, for the inverse operator. For
    this reason it is not the case that :math:`x=g(g^{-1}(x))` during training,
    i.e., that the inverse operation is the inverse of the forward one.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> from pyro.distributions.transforms import AffineAutoregressive
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> iafs = [AffineAutoregressive(AutoRegressiveNN(10, [40])) for _ in range(2)]
    >>> bn = BatchNorm(10)
    >>> flow_dist = dist.TransformedDistribution(base_dist, [iafs[0], bn, iafs[1]])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: the dimension of the input
    :type input_dim: int
    :param momentum: momentum parameter for updating moving averages
    :type momentum: float
    :param epsilon: small number to add to variances to ensure numerical stability
    :type epsilon: float

    References:

    [1] Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep
    Network Training by Reducing Internal Covariate Shift. In International
    Conference on Machine Learning, 2015. https://arxiv.org/abs/1502.03167

    [2] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
    using Real NVP. In International Conference on Learning Representations, 2017.
    https://arxiv.org/abs/1605.08803

    [3] George Papamakarios, Theo Pavlakou, and Iain Murray. Masked Autoregressive
    Flow for Density Estimation. In Neural Information Processing Systems, 2017.
    https://arxiv.org/abs/1705.07057

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 0

    def __init__(self, input_dim, momentum=0.1, epsilon=1e-5):
        super().__init__()

        self.input_dim = input_dim
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.momentum = momentum
        self.epsilon = epsilon

        self.register_buffer('moving_mean', torch.zeros(input_dim))
        self.register_buffer('moving_variance', torch.ones(input_dim))

    @property
    def constrained_gamma(self):
        return F.relu(self.gamma) + 1e-6

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        # Enforcing the constraint that gamma is positive
        return (x - self.beta) / self.constrained_gamma * \
            torch.sqrt(self.moving_variance + self.epsilon) + self.moving_mean

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        # During training, keep smoothed average of sample mean and variance
        if self.training:
            mean, var = y.mean(0), y.var(0)

            with torch.no_grad():
                # NOTE: The momentum variable agrees with the definition in e.g. `torch.nn.BatchNorm1d`
                self.moving_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                self.moving_variance.mul_(1 - self.momentum).add_(var * self.momentum)

        # During test time, use smoothed averages rather than the sample ones
        else:
            mean, var = self.moving_mean, self.moving_variance

        return (y - mean) * self.constrained_gamma / torch.sqrt(var + self.epsilon) + self.beta

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, dx/dy
        """
        if self.training:
            var = torch.var(y, dim=0, keepdim=True)
        else:
            # NOTE: You wouldn't typically run this function in eval mode, but included for gradient tests
            var = self.moving_variance
        return (-self.constrained_gamma.log() + 0.5 * torch.log(var + self.epsilon))


def batchnorm(input_dim, **kwargs):
    """
    A helper function to create a :class:`~pyro.distributions.transforms.BatchNorm`
    object for consistency with other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param momentum: momentum parameter for updating moving averages
    :type momentum: float
    :param epsilon: small number to add to variances to ensure numerical stability
    :type epsilon: float

    """
    bn = BatchNorm(input_dim, **kwargs)
    return bn
