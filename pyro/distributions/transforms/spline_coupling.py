# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.spline import ConditionalSpline, Spline
from pyro.distributions.util import copy_docs_from
from pyro.nn import DenseNN


@copy_docs_from(TransformModule)
class SplineCoupling(TransformModule):
    r"""
    An implementation of the coupling layer with rational spline bijections of
    linear and quadratic order (Durkan et al., 2019; Dolatabadi et al., 2020).
    Rational splines are functions that are comprised of segments that are the ratio
    of two polynomials (see :class:`~pyro.distributions.transforms.Spline`).

    The spline coupling layer uses the transformation,

        :math:`\mathbf{y}_{1:d} = g_\theta(\mathbf{x}_{1:d})`
        :math:`\mathbf{y}_{(d+1):D} = h_\phi(\mathbf{x}_{(d+1):D};\mathbf{x}_{1:d})`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    e.g. :math:`\mathbf{x}_{1:d}` represents the first :math:`d` elements of the
    inputs, :math:`g_\theta` is either the identity function or an elementwise
    rational monotonic spline with parameters :math:`\theta`, and :math:`h_\phi` is
    a conditional elementwise spline spline, conditioning on the first :math:`d`
    elements.

    Example usage:

    >>> from pyro.nn import DenseNN
    >>> input_dim = 10
    >>> split_dim = 6
    >>> count_bins = 8
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [(input_dim - split_dim) * count_bins,
    ... (input_dim - split_dim) * count_bins,
    ... (input_dim - split_dim) * (count_bins - 1),
    ... (input_dim - split_dim) * count_bins]
    >>> hypernet = DenseNN(split_dim, [10*input_dim], param_dims)
    >>> transform = SplineCoupling(input_dim, split_dim, hypernet)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: Dimension of the input vector. Despite operating element-wise,
        this is required so we know how many parameters to store.
    :type input_dim: int
    :param split_dim: Zero-indexed dimension :math:`d` upon which to perform input/
        output split for transformation.
    :param hypernet: a neural network whose forward call returns a tuple of spline
        parameters (see :class:`~pyro.distributions.transforms.ConditionalSpline`).
    :type hypernet: callable
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

    def __init__(self, input_dim, split_dim, hypernet, count_bins=8, bound=3., order='linear', identity=False):
        super(SplineCoupling, self).__init__(cache_size=1)

        # One part of the input is (optionally) put through an element-wise spline and the other part through a
        # conditional one that inputs the first part.
        self.lower_spline = Spline(split_dim, count_bins, bound, order)
        self.upper_spline = ConditionalSpline(hypernet, input_dim - split_dim, count_bins, bound, order)
        self.split_dim = split_dim
        self.identity = identity

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]

        if not self.identity:
            y1 = self.lower_spline(x1)
            log_detK = self.lower_spline._cache_log_detJ
        else:
            y1 = x1

        upper_spline = self.upper_spline.condition(x1)
        y2 = upper_spline(x2)
        log_detJ = upper_spline._cache_log_detJ

        if not self.identity:
            log_detJ = torch.cat([log_detJ, log_detK], dim=-1)
        self._cache_log_detJ = log_detJ

        return torch.cat([y1, y2], dim=-1)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available,
        otherwise performs the inversion afresh.
        """
        y1, y2 = y[..., :self.split_dim], y[..., self.split_dim:]

        if not self.identity:
            x1 = self.lower_spline._inv_call(y1)
            log_detK = self.lower_spline._cache_log_detJ
        else:
            x1 = y1

        upper_spline = self.upper_spline.condition(x1)
        x2 = upper_spline._inv_call(y2)
        log_detJ = upper_spline._cache_log_detJ

        if not self.identity:
            log_detJ = torch.cat([log_detJ, log_detK], dim=-1)
        self._cache_log_detJ = log_detJ

        return torch.cat([x1, x2], dim=-1)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cache_log_detJ.sum(-1)


def spline_coupling(input_dim, split_dim=None, hidden_dims=None, count_bins=8, bound=3.0):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.SplineCoupling` object for consistency
    with other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int

    """

    if split_dim is None:
        split_dim = input_dim // 2

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    nn = DenseNN(split_dim,
                 hidden_dims,
                 param_dims=[(input_dim - split_dim) * count_bins,
                             (input_dim - split_dim) * count_bins,
                             (input_dim - split_dim) * (count_bins - 1),
                             (input_dim - split_dim) * count_bins])

    return SplineCoupling(input_dim, split_dim, nn, count_bins, bound)
