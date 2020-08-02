# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.utils import clamp_preserve_gradients
from pyro.distributions.util import copy_docs_from
from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN


@copy_docs_from(TransformModule)
class AffineAutoregressive(TransformModule):
    r"""
    An implementation of the bijective transform of Inverse Autoregressive Flow
    (IAF), using by default Eq (10) from Kingma Et Al., 2016,

        :math:`\mathbf{y} = \mu_t + \sigma_t\odot\mathbf{x}`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    :math:`\mu_t,\sigma_t` are calculated from an autoregressive network on
    :math:`\mathbf{x}`, and :math:`\sigma_t>0`.

    If the stable keyword argument is set to True then the transformation used is,

        :math:`\mathbf{y} = \sigma_t\odot\mathbf{x} + (1-\sigma_t)\odot\mu_t`

    where :math:`\sigma_t` is restricted to :math:`(0,1)`. This variant of IAF is
    claimed by the authors to be more numerically stable than one using Eq (10),
    although in practice it leads to a restriction on the distributions that can be
    represented, presumably since the input is restricted to rescaling by a number
    on :math:`(0,1)`.

    Together with :class:`~pyro.distributions.TransformedDistribution` this provides
    a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = AffineAutoregressive(AutoRegressiveNN(10, [40]))
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse of the Bijector is required when, e.g., scoring the log density of a
    sample with :class:`~pyro.distributions.TransformedDistribution`. This
    implementation caches the inverse of the Bijector when its forward operation is
    called, e.g., when sampling from
    :class:`~pyro.distributions.TransformedDistribution`. However, if the cached
    value isn't available, either because it was overwritten during sampling a new
    value or an arbitrary value is being scored, it will calculate it manually. Note
    that this is an operation that scales as O(D) where D is the input dimension,
    and so should be avoided for large dimensional uses. So in general, it is cheap
    to sample from IAF and score a value that was sampled by IAF, but expensive to
    score an arbitrary value.

    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns a real-valued mean and logit-scale as a tuple
    :type autoregressive_nn: callable
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float
    :param sigmoid_bias: A term to add the logit of the input when using the stable
        tranform.
    :type sigmoid_bias: float
    :param stable: When true, uses the alternative "stable" version of the transform
        (see above).
    :type stable: bool

    References:

    [1] Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever,
    Max Welling. Improving Variational Inference with Inverse Autoregressive Flow.
    [arXiv:1606.04934]

    [2] Danilo Jimenez Rezende, Shakir Mohamed. Variational Inference with
    Normalizing Flows. [arXiv:1505.05770]

    [3] Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. MADE: Masked
    Autoencoder for Distribution Estimation. [arXiv:1502.03509]

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1
    sign = +1
    autoregressive = True

    def __init__(
            self,
            autoregressive_nn,
            log_scale_min_clip=-5.,
            log_scale_max_clip=3.,
            sigmoid_bias=2.0,
            stable=False
    ):
        super().__init__(cache_size=1)
        self.arn = autoregressive_nn
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.sigmoid_bias = sigmoid_bias
        self.stable = stable

        if stable:
            self._call = self._call_stable
            self._inverse = self._inverse_stable

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
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

        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
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
        Calculates the elementwise determinant of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        elif not self.stable:
            _, log_scale = self.arn(x)
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        else:
            _, logit_scale = self.arn(x)
            log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        return log_scale.sum(-1)

    def _call_stable(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        mean, logit_scale = self.arn(x)
        logit_scale = logit_scale + self.sigmoid_bias
        scale = self.sigmoid(logit_scale)
        log_scale = self.logsigmoid(logit_scale)
        self._cached_log_scale = log_scale

        y = scale * x + (1 - scale) * mean
        return y

    def _inverse_stable(self, y):
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

        self._cached_log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        x = torch.stack(x, dim=-1)
        return x


@copy_docs_from(ConditionalTransformModule)
class ConditionalAffineAutoregressive(ConditionalTransformModule):
    r"""
    An implementation of the bijective transform of Inverse Autoregressive Flow
    (IAF) that conditions on an additional context variable and uses, by default,
    Eq (10) from Kingma Et Al., 2016,

        :math:`\mathbf{y} = \mu_t + \sigma_t\odot\mathbf{x}`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    :math:`\mu_t,\sigma_t` are calculated from an autoregressive network on
    :math:`\mathbf{x}` and context :math:`\mathbf{z}\in\mathbb{R}^M`, and
    :math:`\sigma_t>0`.

    If the stable keyword argument is set to True then the transformation used is,

        :math:`\mathbf{y} = \sigma_t\odot\mathbf{x} + (1-\sigma_t)\odot\mu_t`

    where :math:`\sigma_t` is restricted to :math:`(0,1)`. This variant of IAF is
    claimed by the authors to be more numerically stable than one using Eq (10),
    although in practice it leads to a restriction on the distributions that can be
    represented, presumably since the input is restricted to rescaling by a number
    on :math:`(0,1)`.

    Together with :class:`~pyro.distributions.ConditionalTransformedDistribution`
    this provides a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import ConditionalAutoRegressiveNN
    >>> input_dim = 10
    >>> context_dim = 4
    >>> batch_size = 3
    >>> hidden_dims = [10*input_dim, 10*input_dim]
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hypernet = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims)
    >>> transform = ConditionalAffineAutoregressive(hypernet)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size]))  # doctest: +SKIP

    The inverse of the Bijector is required when, e.g., scoring the log density of a
    sample with :class:`~pyro.distributions.TransformedDistribution`. This
    implementation caches the inverse of the Bijector when its forward operation is
    called, e.g., when sampling from
    :class:`~pyro.distributions.TransformedDistribution`. However, if the cached
    value isn't available, either because it was overwritten during sampling a new
    value or an arbitrary value is being scored, it will calculate it manually. Note
    that this is an operation that scales as O(D) where D is the input dimension,
    and so should be avoided for large dimensional uses. So in general, it is cheap
    to sample from IAF and score a value that was sampled by IAF, but expensive to
    score an arbitrary value.

    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns a real-valued mean and logit-scale as a tuple
    :type autoregressive_nn: nn.Module
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float
    :param sigmoid_bias: A term to add the logit of the input when using the stable
        tranform.
    :type sigmoid_bias: float
    :param stable: When true, uses the alternative "stable" version of the transform
        (see above).
    :type stable: bool

    References:

    [1] Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever,
    Max Welling. Improving Variational Inference with Inverse Autoregressive Flow.
    [arXiv:1606.04934]

    [2] Danilo Jimenez Rezende, Shakir Mohamed. Variational Inference with
    Normalizing Flows. [arXiv:1505.05770]

    [3] Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. MADE: Masked
    Autoencoder for Distribution Estimation. [arXiv:1502.03509]

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, autoregressive_nn, **kwargs):
        super().__init__()
        self.nn = autoregressive_nn
        self.kwargs = kwargs

    def condition(self, context):
        """
        Conditions on a context variable, returning a non-conditional transform of
        of type :class:`~pyro.distributions.transforms.AffineAutoregressive`.
        """

        cond_nn = partial(self.nn, context=context)
        cond_nn.permutation = cond_nn.func.permutation
        cond_nn.get_permutation = cond_nn.func.get_permutation
        return AffineAutoregressive(cond_nn, **self.kwargs)


def affine_autoregressive(input_dim, hidden_dims=None, **kwargs):
    """
    A helper function to create an
    :class:`~pyro.distributions.transforms.AffineAutoregressive` object that takes
    care of constructing an autoregressive network with the correct input/output
    dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param hidden_dims: The desired hidden dimensions of the autoregressive network.
        Defaults to using [3*input_dim + 1]
    :type hidden_dims: list[int]
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float
    :param sigmoid_bias: A term to add the logit of the input when using the stable
        tranform.
    :type sigmoid_bias: float
    :param stable: When true, uses the alternative "stable" version of the transform
        (see above).
    :type stable: bool

    """

    if hidden_dims is None:
        hidden_dims = [3 * input_dim + 1]
    arn = AutoRegressiveNN(input_dim, hidden_dims)
    return AffineAutoregressive(arn, **kwargs)


def conditional_affine_autoregressive(input_dim, context_dim, hidden_dims=None, **kwargs):
    """
    A helper function to create an
    :class:`~pyro.distributions.transforms.ConditionalAffineAutoregressive` object
    that takes care of constructing a dense network with the correct input/output
    dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param context_dim: Dimension of context variable
    :type context_dim: int
    :param hidden_dims: The desired hidden dimensions of the dense network. Defaults
        to using [10*input_dim]
    :type hidden_dims: list[int]
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float
    :param sigmoid_bias: A term to add the logit of the input when using the stable
        tranform.
    :type sigmoid_bias: float
    :param stable: When true, uses the alternative "stable" version of the transform
        (see above).
    :type stable: bool

    """
    if hidden_dims is None:
        hidden_dims = [10 * input_dim]
    nn = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims)
    return ConditionalAffineAutoregressive(nn, **kwargs)
