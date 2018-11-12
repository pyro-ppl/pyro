from __future__ import absolute_import, division, print_function

import math
import numbers
from collections import defaultdict, namedtuple

import torch


class GaussianKDE(object):
    """
    Kernel density estimate with Gaussian kernel.
    """
    def __init__(self, samples, bw_method=None, adjust=0.5):
        self._samples = samples if samples.ndim() == 2 else samples.unsqueeze(-1)  # 1D to 2D
        n = samples.size(0)
        d = samples.size(1) if samples.ndim() == 2 else 1
        if bw_method is None or bw_method == "scott":
            self._bw_method_factor = n ** (-1. / (d + 4))
        elif bw_method == "silverman":
            self._bw_method_factor = (n * (d + 2) / 4.) ** (-1. / (d + 4))
        elif isinstance(bw_method, numbers.Number):
            self._bw_method_factor = bw_method
        else:
            raise ValueError("bw_method should be None, 'scott', "
                             "'silverman', or a scalar.")
        self._bw_factor = adjust * self._bw_method_factor
        samples_centering = self.samples - self.samples.mean()
        cov = samples_centering.t().matmul(samples_centering)
        self._kernel_dist = torch.distributions.MultivariateNormal(self._samples, cov)

    def __call__(self, x, normalize=True):
        x = x if x.ndim() == 2 else x.unsqueeze(-1)  # 1D to 2D
        return self._kernel_dist.log_prob(x.unsqueeze(1)).exp().sum(1)


def resample(input, num_samples, dim=-1, replacement=False):
    """
    Draws `num_samples` samples from `input` at dimension `dim`.
    """
    weights = input.new_ones(input.size(dim))
    indices = torch.multinomial(weights, num_samples, replacement)
    return input.index_select(dim, indices)


def quantile(input, probs, dim=-1):
    """
    Computes quantiles of `input` at `probs`. If `probs` is a scalar,
    the output will be squeezed at `dim`.
    """
    if isinstance(probs, (numbers.Number, list, tuple)):
        probs = input.new_tensor(probs)
    sorted_input = input.sort(dim)[0]
    max_index = input.size(dim) - 1
    indices = probs * max_index

    # because indices is float, we interpolate the quantiles linearly from nearby points
    indices_below = indices.long()
    indices_above = (indices_below + 1).clamp(max=max_index)
    quantiles_above = sorted_input.index_select(dim, indices_above)
    quantiles_below = sorted_input.index_select(dim, indices_below)
    shape_to_broadcast = [-1] * input.ndim()
    shape_to_broadcast[dim] = indices.numel()
    weights_above = (indices - indices_below).reshape(shape_to_broadcast)
    weights_below = 1 - weights_above
    quantiles = weights_below * quantiles_below + weights_above * quantiles_above
    return quantiles if probs.shape != torch.Size([]) else quantiles.squeeze(dim)


def percentile_interval(input, prob, dim=-1):
    """
    Computes percentile interval which assigns equal probability mass
    to each tail of the interval. Here `prob` is the probability mass
    of samples within the interval.
    """
    return quantile(input, [(1 - prob) / 2, (1 + prob) / 2], dim)


def hpdi(input, prob, dim=-1):
    """
    Computes "highest posterior density interval" which is the narrowest
    interval with probability mass `prob`.
    """
    sorted_input = input.sort(dim)[0]
    mass = input.size(dim)
    index_length = int(prob * mass)
    intervals_lower = sorted_input.index_select(dim, torch.arange(mass - index_length))
    intervals_upper = sorted_input.index_select(dim, torch.arange(index_length, mass))
    intervals_length = intervals_upper - intervals_lower

    index_start = intervals_length.argmin(dim)
    indices = torch.stack([index_start, index_start + index_length], dim)
    return torch.gather(sorted_input, dim, indices)


WAICinfo = namedtuple("WAICinfo", ["waic", "p_waic", "se", "waic_vec"])


def _waic_from_batch_log_prob(log_probs):
    n = log_probs.size(0)
    lpd = log_probs.logsumexp(dim=0) - math.log(n)
    p_waic = log_probs.var(0)
    elpd = lpd - p_waic
    waic = -2 * elpd
    se = math.sqrt(n) * waic.std()
    return WAICinfo(waic.sum(), p_waic.sum(), se, waic)


def waic(traces):
    """
    Computes widely applicable/Watanabe–Akaike information criterion (WAIC).

    Reference:

    [1] `Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC`,
    Aki Vehtari, Andrew Gelman, and Jonah Gabry
    """
    trace_log_prob = defaultdict(list)
    for trace in traces:
        obs_nodes = trace.observation_nodes
        trace.compute_log_prob(site_filter=lambda name, site: name in obs_nodes)
        for node in obs_nodes:
            trace_log_prob[node].append(trace.nodes[node]["log_prob"].detach())

    obs_waic_dict = {}
    for node, batch_log_prob in trace_log_prob.items():
        obs_waic_dict[node] = _waic_from_batch_log_prob(torch.stack(batch_log_prob, dim=0))
    return obs_waic_dict


ModelInfo = namedtuple("ModelInfo", ["name", "waic", "p_waic", "se",
                                     "waic_diff", "diff_se", "weight"])


def model_selection(model_traces_dict):
    """
    Returns a list of `ModelInfo` according to its rank based on WAIC information criterion.
    """
    model_waic_dict = {}
    obs = None
    for model, traces in model_traces_dict.items():
        model_waic = waic(traces)
        obs = list(model_waic.keys())[0] if obs is None else obs
        if len(model_waic) > 1 or obs not in model_waic:
            raise ValueError("We can only compare models with the same observation node.")
        model_waic_dict[model] = model_waic[obs]

    model_names = sorted(model_waic_dict)
    waics = torch.stack([model_waic_dict[model].waic for model in model_names])
    _, indices = waics.sort(dim=0)

    waic_diff = waics - waics[indices[0]]
    waic_vecs = torch.stack([model_waic_dict[model].waic_vec for model in model_names])
    diff_se = math.sqrt(waic_vecs.size(1)) * (waic_vecs - waic_vecs[indices[0]]).std()
    weights = torch.nn.functional.softmax(-0.5 * waics)

    model_table = []
    for i in indices:
        name = model_names[i]
        p_waic = model_waic_dict[name].p_waic
        se = model_waic_dict[name].se
        model_table.append(ModelInfo(name, waics[i], p_waic, se,
                                     waic_diff[i], diff_se[i], weights[i]))
    return model_table


def gelman_rubin(input):
    """
    Computes R-hat over chains of samples. The first two dimensions of `input`
    is `C x N` where `C` is the number of chains and `N` is the number of samples.
    """
    C, N = input.size(0), input.size(1)
    chain_mean = input.mean(dim=1)
    var_between = chain_mean.var(dim=0) * C / (C - 1)

    chain_biased_var = input.var(dim=1)
    biased_var_within = chain_biased_var.mean(dim=0)
    var_within = biased_var_within * N / (N - 1)

    var_estimator = biased_var_within + var_between
    return (var_estimator / var_within).sqrt(), var_within, var_estimator


def split_gelman_rubin(input):
    """
    Computes split R-hat over chains of input. The first two dimensions of `input`
    is `C x N` where `C` is the number of chains and `N` is the number of samples.
    """
    C, N_half = input.size(0), input.size(1) // 2
    new_input = torch.stack([input[:, :N_half], input[:, -N_half:]], dim=1).reshape(
        (2 * C, N_half) + input.shape[2:])
    return gelman_rubin(new_input)


def _fft_next_good_size(N):
    # find the smallest number >= N such that the only divisors are 2, 3, 5
    if N <= 2:
        return 2
    while True:
        m = N
        while m % 2 == 0:
            m //= 2
        while m % 3 == 0:
            m //= 3
        while m % 5 == 0:
            m //= 5
        if m == 1:
            return N
        N += 1


def autocorrelation(input):
    """
    Computes the autocorrelation of samples. The last dimension is the number of
    samples. Other dimensions are batch dimensions.

    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    """
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(-1)
    M = _fft_next_good_size(N)
    M2 = 2 * M
    # centering and padding x
    centered_signal = input - input.mean()
    pad = input.new_zeros(input.shape[:-1] + (M2 - N,))
    centered_signal = torch.cat([centered_signal, pad], dim=-1)
    # Fourier transform
    freqvec = torch.rfft(centered_signal, signal_ndim=1, onesided=False)
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1, keepdim=True)
    freqvec_gram = torch.cat([freqvec_gram, input.new_zeros(freqvec_gram.shape)], dim=-1)
    # inverse Fourier transform
    ac = torch.irfft(freqvec_gram, signal_ndim=1, onesided=False)
    ac = ac[..., :N]
    ac = ac / input.new_tensor(range(N, 0, -1))
    ac = ac / ac[..., :1]
    return ac


def autocovariance(input):
    """
    Computes the autocovariance of samples. The last dimension is the number of
    samples. Other dimensions are batch dimensions.
    """
    return autocorrelation(input) * input.var(-1)


def _cummin(input):
    """
    Computes cummulative minimum of input at dimension `dim=0`.
    """
    # FIXME: is there a better trick to find accumulate min of a sequence?
    N = input.size(0)
    input_tril = input.unsqueeze(0).expand((N,) + input.shape)
    triu_mask = input.new_ones(N, N).triu(diagonal=1).reshape((N, N) + (1,) * (input.dim() - 1))
    triu_mask = triu_mask.expand((N, N) + input.shape[1:]) > 0.5
    input_tril[triu_mask] = input.max()
    return input_tril.min(0)


def effective_sample_size(input):
    """
    Computes effective sample size of input. The first two dimensions of `input`
    is `C x N` where `C` is the number of chains and `N` is the number of samples.

    Reference:

    [1] `Introduction to Markov Chain Monte Carlo`,
        Charles J. Geyer
    [2] `Stan Reference Manual version 2.18`,
        Stan Development Team
    """
    C, N = input.size(0), input.size(1)
    # C x N x sample_shape -> C x sample_shape x N
    reshaped_input = input.unsqueeze(-1).transpose(1, -1).reshape((C,) + input.shape[2:] + (N,))
    # find autocovariance for each chain at lag k
    gamma_k_c = autocovariance(reshaped_input)  # C x sample_shape x N
    gamma_k_c = gamma_k_c.unsqueeze(1).transpose(1, -1).reshape((C, N) + input.shape[2:])

    # find autocorrelation at lag k (from Stan reference)
    _, var_within, var_estimator = gelman_rubin(input)  # C x N x sample_shape -> sample_shape
    rho_k = (var_estimator - var_within + gamma_k_c.mean(0)) / var_estimator  # N x sample_shape
    rho_k[0] = 1  # correlation at lag 0 is always 1

    # initial positive sequence (formula 1.18 in [1]) applied for autocorrelation
    Rho_k = rho_k if N % 2 == 0 else rho_k[:-1]
    Rho_k = rho_k.reshape((N // 2, 2) + rho_k.shape[2:]).sum(dim=1)
    # separate the first index
    Rho_init = Rho_k[0]
    # Theoretical, Rho_k is positive, but due to noise of correlation computation,
    # Rho_k might not be positive at some point. So we need to truncate (ignore first index).
    Rho_positive = Rho_k[1:].clamp(min=0)

    # Now we make the initial monotone (decreasing) sequence.
    Rho_monotone = _cummin(Rho_positive)

    # Formula 1.19 in [1]
    tau = -1 + 2 * Rho_init + 2 * Rho_monotone.sum(dim=0)
    return C * N / tau
