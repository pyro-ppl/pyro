# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.distributions.transforms.utils import clamp_preserve_gradients
from pyro.nn import ConditionalDenseNN, DenseNN

from functools import partial


@copy_docs_from(TransformModule)
class AffineCoupling(TransformModule):
    """
    An implementation of the affine coupling layer of RealNVP (Dinh et al., 2017)
    that uses the bijective transform,

        :math:`\\mathbf{y}_{1:d} = \\mathbf{x}_{1:d}`
        :math:`\\mathbf{y}_{(d+1):D} = \\mu + \\sigma\\odot\\mathbf{x}_{(d+1):D}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    e.g. :math:`\\mathbf{x}_{1:d}` represents the first :math:`d` elements of the
    inputs, and :math:`\\mu,\\sigma` are shift and translation parameters calculated
    as the output of a function inputting only :math:`\\mathbf{x}_{1:d}`.

    That is, the first :math:`d` components remain unchanged, and the subsequent
    :math:`D-d` are shifted and translated by a function of the previous components.

    Together with :class:`~pyro.distributions.TransformedDistribution` this provides
    a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import DenseNN
    >>> input_dim = 10
    >>> split_dim = 6
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [input_dim-split_dim, input_dim-split_dim]
    >>> hypernet = DenseNN(split_dim, [10*input_dim], param_dims)
    >>> transform = AffineCoupling(split_dim, hypernet)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse of the Bijector is required when, e.g., scoring the log density of a
    sample with :class:`~pyro.distributions.TransformedDistribution`. This
    implementation caches the inverse of the Bijector when its forward operation is
    called, e.g., when sampling from
    :class:`~pyro.distributions.TransformedDistribution`. However, if the cached
    value isn't available, either because it was overwritten during sampling a new
    value or an arbitary value is being scored, it will calculate it manually.

    This is an operation that scales as O(1), i.e. constant in the input dimension.
    So in general, it is cheap to sample *and* score (an arbitrary value) from
    :class:`~pyro.distributions.transforms.AffineCoupling`.

    :param split_dim: Zero-indexed dimension :math:`d` upon which to perform input/
        output split for transformation.
    :type split_dim: int
    :param hypernet: an autoregressive neural network whose forward call returns a
        real-valued mean and logit-scale as a tuple. The input should have final
        dimension split_dim and the output final dimension input_dim-split_dim for
        each member of the tuple.
    :type hypernet: callable
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float

    References:

    [1] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation
    using Real NVP. ICLR 2017.

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, split_dim, hypernet, log_scale_min_clip=-5., log_scale_max_clip=3.):
        super().__init__(cache_size=1)
        self.split_dim = split_dim
        self.nn = hypernet
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]

        mean, log_scale = self.nn(x1)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale

        y1 = x1
        y2 = torch.exp(log_scale) * x2 + mean
        return torch.cat([y1, y2], dim=-1)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
        """
        y1, y2 = y[..., :self.split_dim], y[..., self.split_dim:]
        x1 = y1
        mean, log_scale = self.nn(x1)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale

        x2 = (y2 - mean) * torch.exp(-log_scale)
        return torch.cat([x1, x2], dim=-1)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        x_old, y_old = self._cached_x_y
        if self._cached_log_scale is not None and x is x_old and y is y_old:
            log_scale = self._cached_log_scale
        else:
            x1 = x[..., :self.split_dim]
            _, log_scale = self.nn(x1)
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        return log_scale.sum(-1)


@copy_docs_from(ConditionalTransformModule)
class ConditionalAffineCoupling(ConditionalTransformModule):
    """
    An implementation of the affine coupling layer of RealNVP (Dinh et al., 2017)
    that conditions on an additional context variable and uses the bijective
    transform,

        :math:`\\mathbf{y}_{1:d} = \\mathbf{x}_{1:d}`
        :math:`\\mathbf{y}_{(d+1):D} = \\mu + \\sigma\\odot\\mathbf{x}_{(d+1):D}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    e.g. :math:`\\mathbf{x}_{1:d}` represents the first :math:`d` elements of the
    inputs, and :math:`\\mu,\\sigma` are shift and translation parameters calculated
    as the output of a function input :math:`\\mathbf{x}_{1:d}` and a context
    variable :math:`\\mathbf{z}\\in\\mathbb{R}^M`.

    That is, the first :math:`d` components remain unchanged, and the subsequent
    :math:`D-d` are shifted and translated by a function of the previous components.

    Together with :class:`~pyro.distributions.ConditionalTransformedDistribution`
    this provides a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import DenseNN
    >>> input_dim = 10
    >>> split_dim = 6
    >>> context_dim = 4
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [input_dim-split_dim, input_dim-split_dim]
    >>> hypernet = ConditionalDenseNN(split_dim, context_dim, [10*input_dim],
    ... param_dims)
    >>> transform = ConditionalAffineCoupling(split_dim, hypernet)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size]))  # doctest: +SKIP

    The inverse of the Bijector is required when, e.g., scoring the log density of a
    sample with :class:`~pyro.distributions.ConditionalTransformedDistribution`.
    This implementation caches the inverse of the Bijector when its forward
    operation is called, e.g., when sampling from
    :class:`~pyro.distributions.ConditionalTransformedDistribution`. However, if the
    cached value isn't available, either because it was overwritten during sampling
    a new value or an arbitary value is being scored, it will calculate it manually.

    This is an operation that scales as O(1), i.e. constant in the input dimension.
    So in general, it is cheap to sample *and* score (an arbitrary value) from
    :class:`~pyro.distributions.transforms.ConditionalAffineCoupling`.

    :param split_dim: Zero-indexed dimension :math:`d` upon which to perform input/
        output split for transformation.
    :type split_dim: int
    :param hypernet: A neural network whose forward call returns a real-valued mean
        and logit-scale as a tuple. The input should have final dimension split_dim
        and the output final dimension input_dim-split_dim for each member of the
        tuple. The network also inputs a context variable as a keyword argument in
        order to condition the output upon it.
    :type hypernet: callable
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the NN
    :type log_scale_max_clip: float

    References:

    Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using
    Real NVP. ICLR 2017.

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, split_dim, hypernet, **kwargs):
        super().__init__()
        self.split_dim = split_dim
        self.nn = hypernet
        self.kwargs = kwargs

    def condition(self, context):
        cond_nn = partial(self.nn, context=context)
        return AffineCoupling(self.split_dim, cond_nn, **self.kwargs)


def affine_coupling(input_dim, hidden_dims=None, split_dim=None, **kwargs):
    """
    A helper function to create an
    :class:`~pyro.distributions.transforms.AffineCoupling` object that takes care of
    constructing a dense network with the correct input/output dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param hidden_dims: The desired hidden dimensions of the dense network. Defaults
        to using [10*input_dim]
    :type hidden_dims: list[int]
    :param split_dim: The dimension to split the input on for the coupling
        transform. Defaults to using input_dim // 2
    :type split_dim: int
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float

    """
    if split_dim is None:
        split_dim = input_dim // 2
    if hidden_dims is None:
        hidden_dims = [10 * input_dim]
    hypernet = DenseNN(split_dim, hidden_dims, [input_dim - split_dim, input_dim - split_dim])
    return AffineCoupling(split_dim, hypernet, **kwargs)


def conditional_affine_coupling(input_dim, context_dim, hidden_dims=None, split_dim=None, **kwargs):
    """
    A helper function to create an
    :class:`~pyro.distributions.transforms.ConditionalAffineCoupling` object that
    takes care of constructing a dense network with the correct input/output
    dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param context_dim: Dimension of context variable
    :type context_dim: int
    :param hidden_dims: The desired hidden dimensions of the dense network. Defaults
        to using [10*input_dim]
    :type hidden_dims: list[int]
    :param split_dim: The dimension to split the input on for the coupling
        transform. Defaults to using input_dim // 2
    :type split_dim: int
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float

    """
    if split_dim is None:
        split_dim = input_dim // 2
    if hidden_dims is None:
        hidden_dims = [10 * input_dim]
    nn = ConditionalDenseNN(split_dim, context_dim, hidden_dims, [input_dim - split_dim, input_dim - split_dim])
    return ConditionalAffineCoupling(split_dim, nn, **kwargs)
