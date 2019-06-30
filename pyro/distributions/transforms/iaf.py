from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.distributions.transforms.utils import clamp_preserve_gradients


@copy_docs_from(TransformModule)
class InverseAutoregressiveFlow(TransformModule):
    """
    An implementation of Inverse Autoregressive Flow, using Eq (10) from Kingma Et Al., 2016,

        :math:`\\mathbf{y} = \\mu_t + \\sigma_t\\odot\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, :math:`\\mu_t,\\sigma_t`
    are calculated from an autoregressive network on :math:`\\mathbf{x}`, and :math:`\\sigma_t>0`.

    Together with `TransformedDistribution` this provides a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> iaf = InverseAutoregressiveFlow(AutoRegressiveNN(10, [40]))
    >>> pyro.module("my_iaf", iaf)  # doctest: +SKIP
    >>> iaf_dist = dist.TransformedDistribution(base_dist, [iaf])
    >>> iaf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse of the Bijector is required when, e.g., scoring the log density of a sample with
    `TransformedDistribution`. This implementation caches the inverse of the Bijector when its forward
    operation is called, e.g., when sampling from `TransformedDistribution`. However, if the cached value
    isn't available, either because it was overwritten during sampling a new value or an arbitary value is
    being scored, it will calculate it manually. Note that this is an operation that scales as O(D) where D is
    the input dimension, and so should be avoided for large dimensional uses. So in general, it is cheap
    to sample from IAF and score a value that was sampled by IAF, but expensive to score an arbitrary value.

    :param autoregressive_nn: an autoregressive neural network whose forward call returns a real-valued
        mean and logit-scale as a tuple
    :type autoregressive_nn: nn.Module
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from the autoregressive NN
    :type log_scale_max_clip: float

    References:

    1. Improving Variational Inference with Inverse Autoregressive Flow [arXiv:1606.04934]
    Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling

    2. Variational Inference with Normalizing Flows [arXiv:1505.05770]
    Danilo Jimenez Rezende, Shakir Mohamed

    3. MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1
    autoregressive = True

    def __init__(self, autoregressive_nn, log_scale_min_clip=-5., log_scale_max_clip=3.):
        super(InverseAutoregressiveFlow, self).__init__(cache_size=1)
        self.arn = autoregressive_nn
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        mean, log_scale = self.arn(x)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale
        scale = torch.exp(log_scale)

        y = scale * x + mean
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise performs the inversion afresh.
        """
        x_size = y.size()[:-1]
        perm = self.arn.permutation
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim

        # NOTE: Inversion is an expensive operation that scales in the dimension of the input
        for idx in perm:
            mean, log_scale = self.arn(torch.stack(x, dim=-1))
            inverse_scale = torch.exp(-clamp_preserve_gradients(
                log_scale[..., idx], min=self.log_scale_min_clip, max=self.log_scale_max_clip))
            mean = mean[..., idx]
            x[idx] = (y[..., idx] - mean) * inverse_scale

        x = torch.stack(x, dim=-1)
        log_scale = clamp_preserve_gradients(log_scale, min=self.log_scale_min_clip, max=self.log_scale_max_clip)
        self._cached_log_scale = log_scale
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        else:
            _, log_scale = self.arn(x)
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        return log_scale.sum(-1)


@copy_docs_from(TransformModule)
class InverseAutoregressiveFlowStable(TransformModule):
    """
    An implementation of an Inverse Autoregressive Flow, using Eqs (13)/(14) from Kingma Et Al., 2016,

        :math:`\\mathbf{y} = \\sigma_t\\odot\\mathbf{x} + (1-\\sigma_t)\\odot\\mu_t`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, :math:`\\mu_t,\\sigma_t`
    are calculated from an autoregressive network on :math:`\\mathbf{x}`, and :math:`\\sigma_t` is
    restricted to :math:`(0,1)`.

    This variant of IAF is claimed by the authors to be more numerically stable than one using Eq (10),
    although in practice it leads to a restriction on the distributions that can be represented,
    presumably since the input is restricted to rescaling by a number on :math:`(0,1)`.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> iaf = InverseAutoregressiveFlowStable(AutoRegressiveNN(10, [40]))
    >>> iaf_module = pyro.module("my_iaf", iaf)
    >>> iaf_dist = dist.TransformedDistribution(base_dist, [iaf])
    >>> iaf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    See `InverseAutoregressiveFlow` docs for a discussion of the running cost.

    :param autoregressive_nn: an autoregressive neural network whose forward call returns a real-valued
        mean and logit-scale as a tuple
    :type autoregressive_nn: nn.Module
    :param sigmoid_bias: bias on the hidden units fed into the sigmoid; default=`2.0`
    :type sigmoid_bias: float

    References:

    1. Improving Variational Inference with Inverse Autoregressive Flow [arXiv:1606.04934]
    Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling

    2. Variational Inference with Normalizing Flows [arXiv:1505.05770]
    Danilo Jimenez Rezende, Shakir Mohamed

    3. MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, autoregressive_nn, sigmoid_bias=2.0):
        super(InverseAutoregressiveFlowStable, self).__init__(cache_size=1)
        self.arn = autoregressive_nn
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.sigmoid_bias = sigmoid_bias
        self._cached_log_scale = None

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        mean, logit_scale = self.arn(x)
        logit_scale = logit_scale + self.sigmoid_bias
        scale = self.sigmoid(logit_scale)
        log_scale = self.logsigmoid(logit_scale)
        self._cached_log_scale = log_scale

        y = scale * x + (1 - scale) * mean
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        x_size = y.size()[:-1]
        perm = self.arn.permutation
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim

        # NOTE: Inversion is an expensive operation that scales in the dimension of the input
        for idx in perm:
            mean, logit_scale = self.arn(torch.stack(x, dim=-1))
            inverse_scale = 1 + torch.exp(-logit_scale[..., idx] - self.sigmoid_bias)
            x[idx] = inverse_scale * y[..., idx] + (1 - inverse_scale) * mean[..., idx]
            self._cached_log_scale = inverse_scale

        x = torch.stack(x, dim=-1)
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        else:
            _, logit_scale = self.arn(x)
            log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        return log_scale.sum(-1)
