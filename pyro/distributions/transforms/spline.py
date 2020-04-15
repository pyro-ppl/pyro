# Copyright Contributors to the Pyro project.
# Copyright (c) 2020 Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie
# Copyright (c) 2019 Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios
# Copyright (c) 2019 Tony Duan
# SPDX-License-Identifier: MIT

# This implementation is adapted in part from:
# * https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py; and,
# * https://github.com/hmdolatabadi/LRS_NF/blob/master/nde/transforms/nonlinearities.py,
# under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from





class SplineLayer(nn.Module):
    """
    Helper class to manage learnable spline. One could imagine this as a standard layer in PyTorch...
    """

    def __init__(self, input_dim, count_bins=8, bound=3., order='linear'):
        super().__init__()

        self.input_dim = input_dim
        self.order = order

        # K rational quadratic segments or 2K rational linear segments...
        self.count_bins = count_bins

        # ...on [-B, B] x [-B, B]
        self.bound = bound

        # Parameters for each dimension
        # TODO: What should the initialization scheme be?
        self.unnormalized_widths = nn.Parameter(torch.randn(self.input_dim, self.count_bins))
        self.unnormalized_heights = nn.Parameter(torch.randn(self.input_dim, self.count_bins))
        self.unnormalized_derivatives = nn.Parameter(torch.randn(self.input_dim, self.count_bins - 1))

        # Rational linear splines have additional lambda parameters
        if self.order == "linear":
            self.unnormalized_lambdas = nn.Parameter(torch.rand(self.input_dim, self.count_bins))
        elif self.order == "quadratic":
            raise ValueError("Monotonic rational quadratic splines not yet implemented!")
        else:
            raise ValueError(
                "Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{}' was found!".format(
                    self.order))

    @property
    def widths(self):
        # widths, unnormalized_widths ~ (input_dim, num_bins)
        return F.softmax(self.unnormalized_widths, dim=-1)

    @property
    def heights(self):
        # heights, unnormalized_heights ~ (input_dim, num_bins)
        return F.softmax(self.unnormalized_heights, dim=-1)

    @property
    def derivatives(self):
        # derivatives, unnormalized_derivatives ~ (input_dim, num_bins-1)
        return F.softplus(self.unnormalized_derivatives)

    @property
    def lambdas(self):
        # lambdas, unnormalized_lambdas ~ (input_dim, num_bins)
        return torch.sigmoid(self.unnormalized_lambdas)

    def __call__(self, x, jacobian=False, **kwargs):
        y, log_detJ = _monotonic_rational_spline(
            x,
            self.widths,
            self.heights,
            self.derivatives,
            self.lambdas,
            bound=self.bound,
            **kwargs)

        if not jacobian:
            return y
        else:
            return y, log_detJ


@copy_docs_from(TransformModule)
class Spline(TransformModule):
    """
    An implementation of the element-wise rational spline bijections of linear
    and quadratic order (Durkan et al., 2019; Dolatabadi et al., 2020).
    Rational splines are functions that are comprised of segments that are the
    ratio of two polynomials. For instance, for the :math:`d`-th dimension and
    the :math:`k`-th segment on the spline, the function will take the form,

        :math:`y_d = \\frac{\\alpha^{(k)}(x_d)}{\\beta^{(k)}(x_d)},`

    where :math:`\\alpha^{(k)}` and :math:`\\beta^{(k)}` are two polynomials of
    order :math:`d`. For :math:`d=1`, we say that the spline is linear, and for
    :math:`d=2`, quadratic. The spline is constructed on the specified bounding
    box, :math:`[-K,K]\\times[-K,K]`, with the identity function used elsewhere
    .

    Rational splines offer an excellent combination of functional flexibility
    whilst maintaining a numerically stable inverse that is of the same
    computational and space complexities as the forward operation. This
    element-wise transform permits the accurate represention of complex
    univariate distributions.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = Spline(10, count_bins=4, bound=3.)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: Dimension of the input vector. Despite operating
        element-wise, this is required so we know how many parameters to store.
    :type input_dim: int
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the
        spline.
    :type order: string

    References:

    Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
    Spline Flows. NeurIPS 2019.

    Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. Invertible Generative
    Modeling using Linear Rational Splines. AISTATS 2020.

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 0

    def __init__(self, input_dim, count_bins=8, bound=3., order='linear'):
        super(Spline, self).__init__(cache_size=1)

        self.layer = SplineLayer(input_dim, count_bins=count_bins, bound=bound, order=order)
        self._cache_log_detJ = None

    def _call(self, x):
        y, log_detJ = self.layer(x, jacobian=True)
        self._cache_log_detJ = log_detJ
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available,
        otherwise performs the inversion afresh.
        """
        x, log_detJ = self.layer(y, jacobian=True, inverse=True)
        self._cache_log_detJ = -log_detJ
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cache_log_detJ


def spline(input_dim, **kwargs):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.Spline` object for consistency with
    other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int

    """

    # TODO: A useful heuristic for choosing number of bins from input
    # dimension like: count_bins=min(5, math.log(input_dim))?
    return Spline(input_dim, **kwargs)
