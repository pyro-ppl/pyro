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

# Searches for which bin an input belongs to (in a way that is parallelizable and amenable to autodiff)


def search_sorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

# Performs gather to select the bin in the correct way on batched inputs


def select_bins(x, idx):
    extra_dims = len(idx.shape) - len(x.shape)
    expanded_x = x.expand(idx.shape[:extra_dims] + (-1,) * len(x.shape))
    return expanded_x.gather(-1, idx).squeeze(-1)

# Calculating a monotonic rational (linear for now) spline or its inverse, plus the log(abs(detJ))


def monotonic_rational_spline(inputs, unnormalized_widths, unnormalized_heights,
                              unnormalized_derivatives, unnormalized_lambdas, inverse=False,
                              left=0.,
                              right=1.,
                              bottom=0.,
                              top=1.,
                              min_bin_width=1e-3,
                              min_bin_height=1e-3,
                              min_derivative=1e-3):

    num_bins = unnormalized_widths.shape[-1]
    # if min_bin_width * num_bins > 1.0:
    #    raise ValueError('Minimal bin width too large for the number of bins')
    # if min_bin_height * num_bins > 1.0:
    #    raise ValueError('Minimal bin height too large for the number of bins')

    # inputs, inside_interval_mask, outside_interval_mask ~ (batch_dim, input_dim)
    inside_interval_mask = (inputs >= left) & (inputs <= right)
    outside_interval_mask = ~inside_interval_mask

    # outputs, logabsdet ~ (batch_dim, input_dim)
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # Normalize widths
    # widths, unnormalized_widths ~ (input_dim, num_bins)
    widths = F.softmax(unnormalized_widths, dim=-1)

    # Give every bin min_bin_width, then add width fraction of remaining length
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    # Cumulative widths gives x (y for inverse) position of knots
    cumwidths = torch.cumsum(widths, dim=-1)

    # Pad left of last dimension with 1 zero
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

    # Translate [0,1] knot points to [-B, B]
    cumwidths = (right - left) * cumwidths + left

    # Are following necessary?
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # Do the same for the heights (make a function for this?)
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights

    # unnormalized_heights ~ (input_dim, num_bins)
    # unnormalized_derivatives ~ (input_dim, num_bins-1)
    # unnormalized_lambdas ~ (input_dim, num_bins)
    # TODO: Comment on purpose of the following? => so when normalized by softplus...
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    # Normalize derivatives
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    # TODO: I think I can remove padding, and simplify searchsorted
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # Get the index of the bin that each input is in
    bin_idx = search_sorted(cumheights if inverse else cumwidths, inputs)[..., None]

    # The x_k for each input's bin
    # cum_widths ~ (input_dim, num_bins+1)
    # bin_idx ~ (batch_dim, input_dim, 1)
    input_cumwidths = select_bins(cumwidths, bin_idx)

    # x_{k+1} - x_{k}
    # widths ~ (input_dim, num_bins)
    input_bin_widths = select_bins(widths, bin_idx)

    # y_k for each input's bin
    # cumheights ~ (input_dim, num_bins+1)
    input_cumheights = select_bins(cumheights, bin_idx)

    # (y_{k+1} - y_{k}) / (x_{k+1} - x_{k}) for each input's bin
    # widths ~ (input_dim, num_bins)
    delta = heights / widths
    input_delta = select_bins(delta, bin_idx)

    # delta_k, derivative at knot starting bin for each input
    # widths ~ (input_dim, num_bins+1)
    input_derivatives = select_bins(derivatives, bin_idx)

    # delta_{k+1}, derivative at knot ending bin for each input
    # widths ~ (input_dim, num_bins)
    input_derivatives_plus_one = select_bins(derivatives[..., 1:], bin_idx)

    # s_k = (y_{k+1} - y_{k}) for each input's bin
    # widths ~ (input_dim, num_bins)
    input_heights = select_bins(heights, bin_idx)

    # For numerical stability, restrict 0.025 <= lambda <= 0.975
    # widths ~ (input_dim, num_bins)
    lambdas = 0.95 * torch.sigmoid(unnormalized_lambdas) + 0.025

    # Select the lambda for each input's bin
    lam = select_bins(lambdas, bin_idx)

    # The weight, w_a, at the left-hand-side of each bin
    # We are free to choose w_a, so set it to 1
    wa = 1.0

    # The weight, w_b, at the right-hand-side of each bin
    # This turns out to be a multiple of the w_a
    # TODO: Should this be done in log space?
    wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa

    # The weight, w_c, at the division point of each bin
    # Recall that each bin is divided into two parts so we have enough d.o.f. to fit spline
    wc = (lam * wa * input_derivatives + (1 - lam) * wb * input_derivatives_plus_one) / input_delta

    # Calculate y coords of bins
    ya = input_cumheights
    yb = input_heights + input_cumheights
    yc = ((1.0 - lam) * wa * ya + lam * wb * yb) / ((1.0 - lam) * wa + lam * wb)

    # TODO: Work around this quick hack
    inputs[outside_interval_mask] = 0.0

    # The core monotonic rational spline equation
    if inverse:
        # TODO: Comments for inverse calculation
        numerator = (lam * wa * (ya - inputs)) * (inputs <= yc).float() \
            + ((wc - lam * wb) * inputs + lam * wb * yb - wc * yc) * (inputs > yc).float()

        denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (inputs <= yc).float()\
            + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

        theta = numerator / denominator

        outputs = theta * input_bin_widths + input_cumwidths

        derivative_numerator = (wa * wc * lam * (yc - ya) * (inputs <= yc).float()
                                + wb * wc * (1 - lam) * (yb - yc) * (inputs > yc).float()) * input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))

    else:
        # TODO: Comments!
        # phi in notation of paper
        theta = (inputs - input_cumwidths) / input_bin_widths

        numerator = (wa * ya * (lam - theta) + wc * yc * theta) * (theta <= lam).float()\
            + (wc * yc * (1 - theta) + wb * yb * (theta - lam)) * (theta > lam).float()

        denominator = (wa * (lam - theta) + wc * theta) * (theta <= lam).float()\
            + (wc * (1 - theta) + wb * (theta - lam)) * (theta > lam).float()

        outputs = numerator / denominator

        derivative_numerator = (wa * wc * lam * (yc - ya) * (theta <= lam).float()
                                + wb * wc * (1 - lam) * (yb - yc) * (theta > lam).float()) / input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))

    # Apply the identity function outside the bounding box
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0
    return outputs, logabsdet


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

    def __init__(self, input_dim, count_bins=5, bounding_box=3., order='linear'):
        super(Spline, self).__init__(cache_size=1)
        self.input_dim = input_dim
        self.order = order

        # K rational quadratic segments, 2K rational linear segments...
        self.K = count_bins

        # ...on [-B, B] x [-B, B]
        self.B = bounding_box

        # Parameters for each dimension
        # TODO: What should the initialization scheme be?
        self.unnormalized_widths = nn.Parameter(torch.randn(self.input_dim, self.K))
        self.unnormalized_heights = nn.Parameter(torch.randn(self.input_dim, self.K))
        self.unnormalized_derivatives = nn.Parameter(torch.randn(self.input_dim, self.K - 1))

        # Rational linear splines have additional lambda parameters
        if self.order == "linear":
            self.unnormalized_lambdas = nn.Parameter(torch.rand(self.input_dim, self.K))
        elif self.order != "quadratic":
            raise ValueError(
                f"Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{self.order}' was found!")

        # Store the log determinate in this!
        self._cache_log_detJ = None

    def _call(self, x):
        y, log_detJ = monotonic_rational_spline(
                        x,
                        self.unnormalized_widths,
                        self.unnormalized_heights,
                        self.unnormalized_derivatives,
                        self.unnormalized_lambdas,
                        inverse=False,
                        left=-self.B, right=self.B, bottom=-self.B, top=self.B)
        self._cache_log_detJ = log_detJ

        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise performs the inversion afresh.
        """
        x, log_detJ = monotonic_rational_spline(
                        y,
                        self.unnormalized_widths,
                        self.unnormalized_heights,
                        self.unnormalized_derivatives,
                        self.unnormalized_lambdas,
                        inverse=True,
                        left=-self.B, right=self.B, bottom=-self.B, top=self.B)
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

    # TODO: A useful heuristic for choosing number of bins from input dimension like: count_bins=min(5, math.log(input_dim))?
    return Spline(input_dim, **kwargs)

import torch.distributions as dist
def _test_inverse(input_dim, transform):
    base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))

    x_true = base_dist.sample(torch.Size([10]))
    y = transform._call(x_true)

    # Cache is empty, hence must be calculating inverse afresh
    x_calculated = transform._inverse(y)

    assert torch.norm(x_true - x_calculated, dim=-1).max().item() < 1e-4

# TODO: Remove the following and create a Pyro tests!
if __name__ == "__main__":
    #sc = spline(input_dim=3)
    #x = torch.randn(1,3)
    #y = sc._call(x)
    #x2 = sc._inverse(y)
    #print(x - x2)
    _test_inverse(5, spline(5))
