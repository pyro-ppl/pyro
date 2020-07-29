# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch
from torch.distributions import constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.spline import ConditionalSpline
from pyro.distributions.util import copy_docs_from
from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN


@copy_docs_from(TransformModule)
class SplineAutoregressive(TransformModule):
    r"""
    An implementation of the autoregressive layer with rational spline bijections of
    linear and quadratic order (Durkan et al., 2019; Dolatabadi et al., 2020).
    Rational splines are functions that are comprised of segments that are the ratio
    of two polynomials (see :class:`~pyro.distributions.transforms.Spline`).

    The autoregressive layer uses the transformation,

        :math:`y_d = g_{\theta_d}(x_d)\ \ \ d=1,2,\ldots,D`

    where :math:`\mathbf{x}=(x_1,x_2,\ldots,x_D)` are the inputs,
    :math:`\mathbf{y}=(y_1,y_2,\ldots,y_D)` are the outputs, :math:`g_{\theta_d}` is
    an elementwise rational monotonic spline with parameters :math:`\theta_d`, and
    :math:`\theta=(\theta_1,\theta_2,\ldots,\theta_D)` is the output of an
    autoregressive NN inputting :math:`\mathbf{x}`.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> input_dim = 10
    >>> count_bins = 8
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hidden_dims = [input_dim * 10, input_dim * 10]
    >>> param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    >>> hypernet = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims)
    >>> transform = SplineAutoregressive(input_dim, hypernet, count_bins=count_bins)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: Dimension of the input vector. Despite operating element-wise,
        this is required so we know how many parameters to store.
    :type input_dim: int
    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns tuple of the spline parameters
    :type autoregressive_nn: callable
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    References:

    Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
    Spline Flows. NeurIPS 2019.

    Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. Invertible Generative
    Modeling using Linear Rational Splines. AISTATS 2020.

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1
    autoregressive = True

    def __init__(
            self,
            input_dim,
            autoregressive_nn,
            count_bins=8,
            bound=3.,
            order='linear'
    ):
        super(SplineAutoregressive, self).__init__(cache_size=1)
        self.arn = autoregressive_nn
        self.spline = ConditionalSpline(autoregressive_nn, input_dim, count_bins, bound, order)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        spline = self.spline.condition(x)
        y = spline(x)
        self._cache_log_detJ = spline._cache_log_detJ
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
        """
        input_dim = y.size(-1)
        x = torch.zeros_like(y)

        # NOTE: Inversion is an expensive operation that scales in the dimension of the input
        for _ in range(input_dim):
            spline = self.spline.condition(x)
            x = spline._inverse(y)

        self._cache_log_detJ = spline._cache_log_detJ
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

        return self._cache_log_detJ.sum(-1)


@copy_docs_from(ConditionalTransformModule)
class ConditionalSplineAutoregressive(ConditionalTransformModule):
    r"""
    An implementation of the autoregressive layer with rational spline bijections of
    linear and quadratic order (Durkan et al., 2019; Dolatabadi et al., 2020) that
    conditions on an additional context variable. Rational splines are functions
    that are comprised of segments that are the ratio of two polynomials (see
    :class:`~pyro.distributions.transforms.Spline`).

    The autoregressive layer uses the transformation,

        :math:`y_d = g_{\theta_d}(x_d)\ \ \ d=1,2,\ldots,D`

    where :math:`\mathbf{x}=(x_1,x_2,\ldots,x_D)` are the inputs,
    :math:`\mathbf{y}=(y_1,y_2,\ldots,y_D)` are the outputs, :math:`g_{\theta_d}` is
    an elementwise rational monotonic spline with parameters :math:`\theta_d`, and
    :math:`\theta=(\theta_1,\theta_2,\ldots,\theta_D)` is the output of a
    conditional autoregressive NN inputting :math:`\mathbf{x}` and conditioning on
    the context variable :math:`\mathbf{z}`.

    Example usage:

    >>> from pyro.nn import ConditionalAutoRegressiveNN
    >>> input_dim = 10
    >>> count_bins = 8
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hidden_dims = [input_dim * 10, input_dim * 10]
    >>> param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    >>> hypernet = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims,
    ... param_dims=param_dims)
    >>> transform = ConditionalSplineAutoregressive(input_dim, hypernet,
    ... count_bins=count_bins)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size]))  # doctest: +SKIP

    :param input_dim: Dimension of the input vector. Despite operating element-wise,
        this is required so we know how many parameters to store.
    :type input_dim: int
    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns tuple of the spline parameters
    :type autoregressive_nn: callable
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    References:

    Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
    Spline Flows. NeurIPS 2019.

    Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. Invertible Generative
    Modeling using Linear Rational Splines. AISTATS 2020.

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, input_dim, autoregressive_nn, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.nn = autoregressive_nn
        self.kwargs = kwargs

    def condition(self, context):
        """
        Conditions on a context variable, returning a non-conditional transform of
        of type :class:`~pyro.distributions.transforms.SplineAutoregressive`.
        """

        # Note that nn.condition doesn't copy the weights of the ConditionalAutoregressiveNN
        cond_nn = partial(self.nn, context=context)
        cond_nn.permutation = cond_nn.func.permutation
        cond_nn.get_permutation = cond_nn.func.get_permutation
        return SplineAutoregressive(self.input_dim, cond_nn, **self.kwargs)


def spline_autoregressive(input_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear'):
    r"""
    A helper function to create an
    :class:`~pyro.distributions.transforms.SplineAutoregressive` object that takes
    care of constructing an autoregressive network with the correct input/output
    dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param hidden_dims: The desired hidden dimensions of the autoregressive network.
        Defaults to using [3*input_dim + 1]
    :type hidden_dims: list[int]
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    """

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims)
    return SplineAutoregressive(input_dim, arn, count_bins=count_bins, bound=bound, order=order)


def conditional_spline_autoregressive(input_dim, context_dim, hidden_dims=None, count_bins=8, bound=3.0,
                                      order='linear'):
    r"""
    A helper function to create a
    :class:`~pyro.distributions.transforms.ConditionalSplineAutoregressive` object
    that takes care of constructing an autoregressive network with the correct
    input/output dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param context_dim: Dimension of context variable
    :type context_dim: int
    :param hidden_dims: The desired hidden dimensions of the autoregressive network.
        Defaults to using [input_dim * 10, input_dim * 10]
    :type hidden_dims: list[int]
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    """

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    arn = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims, param_dims=param_dims)
    return ConditionalSplineAutoregressive(input_dim, arn, count_bins=count_bins, bound=bound, order=order)
