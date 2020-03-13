# This implementation is adapted in part from:
# * https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py; and,
# * https://github.com/hmdolatabadi/LRS_NF/blob/master/nde/transforms/nonlinearities.py,
# under the MIT license.
import torch
import torch.nn as nn
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import constraints
import torch.nn.functional as F

import numpy as np

from pyro.distributions.util import copy_docs_from

# TODO: Move the spline functions to a pyro.numerics or torch.numerics module?
# This would be a good place for e.g. numerical ODE solvers as well...

def search_sorted(bin_locations, inputs, eps=1e-6):
    # Searches for which bin an input belongs to (in a way that is parallelizable and amenable to autodiff)
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def select_bins(x, idx):
    # Performs gather to select the bin in the correct way on batched inputs
    idx = idx.clamp(min=0, max=x.size(-1) - 1)
    extra_dims = len(idx.shape) - len(x.shape)
    expanded_x = x.expand(idx.shape[:extra_dims] + (-1,) * len(x.shape))
    return expanded_x.gather(-1, idx).squeeze(-1)


def monotonic_rational_spline(inputs, widths, heights, derivatives, lambdas,
                              inverse=False,
                              bound=3.,
                              min_bin_width=1e-3,
                              min_bin_height=1e-3,
                              min_derivative=1e-3,
                              min_lambda=0.025):
    # Calculating a monotonic rational spline (linear for now) or its inverse, plus the log(abs(detJ)) required
    # for normalizing flows

    # Ensure bound is positive
    # NOTE: For simplicity, we apply the identity function outside [-B, B] X [-B, B] rather than allowing arbitrary
    # corners to the bounding box. If you want a different bounding box you can apply an affine transform before and
    # after the input
    assert bound > 0.0

    num_bins = widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    # inputs, inside_interval_mask, outside_interval_mask ~ (batch_dim, input_dim)
    left, right = -bound, bound
    bottom, top = -bound, bound
    inside_interval_mask = (inputs >= left) & (inputs <= right)
    outside_interval_mask = ~inside_interval_mask

    # outputs, logabsdet ~ (batch_dim, input_dim)
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # For numerical stability, put lower/upper limits on parameters. E.g .give every bin min_bin_width,
    # then add width fraction of remaining length
    # NOTE: Do this here rather than higher up because we want everything to ensure numerical
    # stability within this function
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    derivatives = min_derivative + derivatives
    lambdas = (1-2*min_lambda) * lambdas + min_lambda
    
    # *** TODO: Make a helper function so avoid code repeat for calculating cumwidths etc. ***

    # Cumulative widths gives x (y for inverse) position of knots
    cumwidths = torch.cumsum(widths, dim=-1)

    # Pad left of last dimension with 1 zero
    # TODO: Why???
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

    # Translate [0,1] knot points to [-B, B]
    cumwidths = (right - left) * cumwidths + left

    # Are following necessary?
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # TODO: I think I can remove padding, and simplify searchsorted
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # Pad left and right derivatives with fixed values at first and last knots
    # These are 1 since the function is the identity outside the bounding box and the derivative is continuous
    # NOTE: Not sure why this is 1.0 - min_derivative rather than 1.0. I've copied this from original implementation
    derivatives = F.pad(derivatives, pad=(1, 1), mode='constant', value=1.0-min_derivative)

    # Get the index of the bin that each input is in
    # bin_idx ~ (batch_dim, input_dim, 1)
    bin_idx = search_sorted(cumheights if inverse else cumwidths, inputs)[..., None]

    # Select the value for the relevant bin for the variables used in the main calculation
    # cum_widths ~ (input_dim, num_bins+1)
    input_widths = select_bins(widths, bin_idx)
    input_cumwidths = select_bins(cumwidths, bin_idx)
    input_cumheights = select_bins(cumheights, bin_idx)
    input_delta = select_bins(heights/widths, bin_idx)
    input_derivatives = select_bins(derivatives, bin_idx)
    input_derivatives_plus_one = select_bins(derivatives[..., 1:], bin_idx)
    input_heights = select_bins(heights, bin_idx)
    input_lambdas = select_bins(lambdas, bin_idx)

    # The weight, w_a, at the left-hand-side of each bin
    # We are free to choose w_a, so set it to 1
    wa = 1.0

    # The weight, w_b, at the right-hand-side of each bin
    # This turns out to be a multiple of the w_a
    # TODO: Should this be done in log space for numerical stability?
    wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa

    # The weight, w_c, at the division point of each bin
    # Recall that each bin is divided into two parts so we have enough d.o.f. to fit spline
    wc = (input_lambdas * wa * input_derivatives + (1 - input_lambdas) * wb * input_derivatives_plus_one) / input_delta

    # Calculate y coords of bins
    ya = input_cumheights
    yb = input_heights + input_cumheights
    yc = ((1.0 - input_lambdas) * wa * ya + input_lambdas * wb * yb) / ((1.0 - input_lambdas) * wa + input_lambdas * wb)

    # The core monotonic rational spline equation
    if inverse:
        # TODO: Comments for inverse calculation
        numerator = (input_lambdas * wa * (ya - inputs)) * (inputs <= yc).float() \
            + ((wc - input_lambdas * wb) * inputs + input_lambdas * wb * yb - wc * yc) * (inputs > yc).float()

        denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (inputs <= yc).float()\
            + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

        theta = numerator / denominator

        outputs = theta * input_widths + input_cumwidths

        derivative_numerator = (wa * wc * input_lambdas * (yc - ya) * (inputs <= yc).float()
                                + wb * wc * (1 - input_lambdas) * (yb - yc) * (inputs > yc).float()) * input_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))

    else:
        # TODO: Comments!
        # phi in notation of paper
        theta = (inputs - input_cumwidths) / input_widths

        numerator = (wa * ya * (input_lambdas - theta) + wc * yc * theta) * (theta <= input_lambdas).float()\
            + (wc * yc * (1 - theta) + wb * yb * (theta - input_lambdas)) * (theta > input_lambdas).float()

        denominator = (wa * (input_lambdas - theta) + wc * theta) * (theta <= input_lambdas).float()\
            + (wc * (1 - theta) + wb * (theta - input_lambdas)) * (theta > input_lambdas).float()

        outputs = numerator / denominator

        derivative_numerator = (wa * wc * input_lambdas * (yc - ya) * (theta <= input_lambdas).float()
                                + wb * wc * (1 - input_lambdas) * (yb - yc) * (theta > input_lambdas).float()) / input_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))

    # Apply the identity function outside the bounding box
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0.0
    return outputs, logabsdet


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
                f"Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{self.order}' was found!")

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
        # TODO: Dims?
        return F.softplus(self.unnormalized_derivatives)

    @property
    def lambdas(self):
        # TODO: Dims?
        return torch.sigmoid(self.unnormalized_lambdas)
    
    def __call__(self, x, jacobian=False, **kwargs):
        y, log_detJ = monotonic_rational_spline(
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
    An implementation of the element-wise rational spline bijections of linear and quadratic order.

    References:

    Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural Spline Flows. NeurIPS 2019.

    Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. Invertible Generative Modeling using Linear
    Rational Splines. AISTATS 2020.

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 0

    def __init__(self, *args, **kwargs):
        # TODO: Better way to do this???
        # The problem is essentially how to initialize multiple parent classes with different signatures
        super(Spline, self).__init__(cache_size=1)

        self.layer = SplineLayer(*args, **kwargs)
        self._cache_log_detJ = None

    def _call(self, x):
        y, log_detJ = self.layer(x, jacobian=True)
        self._cache_log_detJ = log_detJ
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise performs the inversion afresh.
        """
        x, log_detJ = self.layer(y, jacobian=True, inverse=True)

        # TODO: Test the following line: should it have a negative sign???
        self._cache_log_detJ = -log_detJ
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        return self._cache_log_detJ


def spline(input_dim, **kwargs):
    """
    A helper function to create a :class:`~pyro.distributions.transforms.Spline` object for consistency with other
    helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int

    """

    # TODO: A useful heuristic for choosing number of bins from input
    # dimension like: count_bins=min(5, math.log(input_dim))?
    return Spline(input_dim, **kwargs)
