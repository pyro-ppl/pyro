from __future__ import absolute_import, division, print_function

from numbers import Number

import torch


class GaussianKDE(object):
    """
    Kernel density estimate with Gaussian kernel.
    """
    def __init__(self, samples, bw_method=None, adjust=0.5):
        self.samples = samples
        if bw_method is None:
            n = samples.size(0)
            d = samples.size(1) if samples.ndim() == 2 else 1
            if bw_method is None or bw_method == "scott":
                self._bw_method_factor = n ** (-1. / (d + 4))
            elif bw_method == "silverman":
                self._bw_method_factor = (n * (d + 2) / 4.) ** (-1. / (d + 4))
            elif isinstance(bw_method, Number):
                self._bw_method_factor = bw_method
            else:
                raise ValueError("bw_method should be None, 'scott', "
                                 "'silverman', or a scalar.")
        self.adjust(adjust)

    def adjust(self, adjust=0.5):
        self.bw_factor = adjust * self._bw_method_factor

    def __call__(self, x, normalize=True):
        # FIXME deal with 2D case
        y = torch.distributions.Normal(self.samples, self.bw_factor).log_prob(
            x.unsqueeze(-1)).logsumexp(-1).exp()
        y = y / y.sum() * (x.size(0) / (self.samples.max() - self.samples.min()))
        return y


def quantile(samples, probs, dim=-1):
    """
    Computes quantiles of `samples` at `probs`. If `probs` is a scalar,
    the output will be squeezed at `dim`.
    """
    if isinstance(probs, (Number, list, tuple)):
        probs = samples.new_tensor(probs)
    sorted_samples = samples.sort(dim)[0]
    max_index = samples.size(dim) - 1
    indices = probs * max_index
    indices_below = indices.long()
    indices_above = (indices_below + 1).clamp(max=max_index)
    quantiles_above = sorted_samples.index_select(dim, indices_above)
    quantiles_below = sorted_samples.index_select(dim, indices_below)
    shape_to_broadcast = [-1] * samples.ndim()
    shape_to_broadcast[dim] = masses.numel()
    weights_above = (masses - indices_below).reshape(shape_to_broadcast)
    weights_below = 1 - weights_above
    quantiles = weights_below * quantiles_below + weights_above * quantiles_above
    return quantiles if probs.shape != torch.Size([]) else quantiles.squeeze(dim)


def percentile_interval(samples, prob, dim=-1):
    """
    Computes percentile interval which assigns equal probability mass
    to each tail of the interval. Here `prob` is the probability mass
    of samples within the interval.
    """
    return quantile(samples, [(1 - prob) / 2, (1 + prob) / 2], dim)


def hpdi(samples, prob, dim=-1):
    """
    Computes "highest posterior density interval" which is the narrowest
    interval with probability mass `prob`.
    """
    sorted_samples = samples.sort(dim)[0]
    mass = samples.size(dim)
    index_length = int(prob * mass)
    intervals_lower = sorted_samples.index_select(dim, torch.arange(mass - index_length))
    intervals_upper = sorted_samples.index_select(dim, torch.arange(index_length, mass))
    intervals_length = intervals_upper - intervals_lower
    index_start = intervals_length.argmin(dim)
    indices = torch.stack([index_start, index_start + index_length], dim)
    return torch.gather(sorted_samples, dim, indices)


def waic(log_likehoods):
    """widely applicable information criterion"""
    pass


def gelman_rubin(samples, dim=0):
    """Compute R-hat over chains of samples. Chain dimension is specified by `dim`."""
    pass


def split_gelman_rubin(samples, dim=0):
    """Compute split R-hat over chains of samples. Chain dimension is specified by `dim`."""
    pass


def effective_number(samples, dim=0):
    """Compute effective number of samples"""
    pass
