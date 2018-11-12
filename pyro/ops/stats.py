from __future__ import absolute_import, division, print_function

import torch


def _compute_chain_variance_stats(input):
    # compute within-chain variance and variance estimator
    N = input.size(1)
    chain_mean = input.mean(dim=1)
    var_between = chain_mean.var(dim=0)

    chain_var = input.var(dim=1)
    var_within = chain_var.mean(dim=0)
    var_estimator = (N - 1) / N * var_within + var_between
    return var_within, var_estimator


def gelman_rubin(input):
    """
    Computes R-hat over chains of samples. The first two dimensions of `input`
    is `C x N` where `C` is the number of chains and `N` is the number of samples.
    It is required that `C >= 2` and `N >= 2`.

    :param torch.Tensor input: the input tensor.
    :returns torch.Tensor: R-hat of `input`.
    """
    assert input.dim() >= 2
    assert input.size(0) >= 2
    assert input.size(1) >= 2
    var_within, var_estimator = _compute_chain_variance_stats(input)
    return (var_estimator / var_within).sqrt()


def split_gelman_rubin(input):
    """
    Computes R-hat over chains of samples. The first two dimensions of `input`
    is `C x N` where `C` is the number of chains and `N` is the number of samples.
    It is required that `C >= 2` and `N >= 4`.

    :param torch.Tensor input: the input tensor.
    :returns torch.Tensor: split R-hat of `input`.
    """
    assert input.dim() >= 2
    assert input.size(1) >= 4
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

    :param torch.Tensor input: the input tensor.
    :returns torch.Tensor: autocorrelation of `input`.
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

    :param torch.Tensor input: the input tensor.
    :returns torch.Tensor: autocorrelation of `input`.
    """
    return autocorrelation(input) * input.var(-1, unbiased=False, keepdim=True)


def _cummin(input):
    """
    Computes cummulative minimum of input at dimension `dim=0`.
    """
    # FIXME: is there a better trick to find accumulate min of a sequence?
    N = input.size(0)
    input_tril = input.unsqueeze(0).repeat((N,) + (1,) * input.dim())
    triu_mask = input.new_ones(N, N).triu(diagonal=1).reshape((N, N) + (1,) * (input.dim() - 1))
    triu_mask = triu_mask.expand((N, N) + input.shape[1:]) > 0.5
    input_tril.masked_fill_(triu_mask, input.max())
    return input_tril.min(dim=1)[0]


def effective_sample_size(input):
    """
    Computes effective sample size of input. The first two dimensions of `input`
    is `C x N` where `C` is the number of chains and `N` is the number of samples.

    Reference:
    [1] `Introduction to Markov Chain Monte Carlo`,
        Charles J. Geyer
    [2] `Stan Reference Manual version 2.18`,
        Stan Development Team

    :param torch.Tensor input: the input tensor.
    :returns torch.Tensor: effective sample size of `input`.
    """
    C, N = input.size(0), input.size(1)
    # C x N x sample_shape -> C x sample_shape x N
    reshaped_input = input.unsqueeze(-1).transpose(1, -1).reshape((C,) + input.shape[2:] + (N,))
    # find autocovariance for each chain at lag k
    gamma_k_c = autocovariance(reshaped_input)  # C x sample_shape x N
    gamma_k_c = gamma_k_c.unsqueeze(1).transpose(1, -1).reshape((C, N) + input.shape[2:])
    # find autocorrelation at lag k (from Stan reference)
    var_within, var_estimator = _compute_chain_variance_stats(input)  # sample_shape
    rho_k = (var_estimator - var_within + gamma_k_c.mean(0)) / var_estimator  # N x sample_shape
    rho_k[0] = 1  # correlation at lag 0 is always 1
    # initial positive sequence (formula 1.18 in [1]) applied for autocorrelation
    Rho_k = rho_k if N % 2 == 0 else rho_k[:-1]
    Rho_k = Rho_k.reshape((N // 2, 2) + rho_k.shape[2:]).sum(dim=1)
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
