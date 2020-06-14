# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.spline import ConditionalSpline
from pyro.distributions.transforms.utils import clamp_preserve_gradients
from pyro.distributions.util import copy_docs_from
from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN


@copy_docs_from(TransformModule)
class SplineAutoregressive(TransformModule):
    r"""
    TODO

    """

    domain = constraints.real
    codomain = constraints.real
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
    TODO

    """

    domain = constraints.real
    codomain = constraints.real
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


def spline_autoregressive(input_dim, hidden_dims=None, count_bins=8, bound=3.0):
    """
    A helper function to create an
    :class:`~pyro.distributions.transforms.SplineAutoregressive` object that takes
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
        hidden_dims = [input_dim * 10, input_dim * 10]

    param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims)
    return SplineAutoregressive(input_dim, arn, count_bins=count_bins, bound=bound, order='linear')


def conditional_spline_autoregressive(input_dim, context_dim, hidden_dims=None, **kwargs):
    """
    A helper function to create an
    :class:`~pyro.distributions.transforms.ConditionalSplineAutoregressive` object
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
    permutation = torch.arange(input_dim, device='cpu').to(torch.Tensor().device)
    nn = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims, permutation=permutation)
    return ConditionalSplineAutoregressive(nn, **kwargs)
    