from __future__ import absolute_import, division, print_function

import torch


def _compute_chain_variance_stats(input):
    # compute within-chain variance and variance estimator
    # input has shape N x C x sample_shape
    N = input.size(0)
    chain_mean = input.mean(dim=0)
    var_between = chain_mean.var(dim=0)

    chain_var = input.var(dim=0)
    var_within = chain_var.mean(dim=0)
    var_estimator = (N - 1) / N * var_within + var_between
    return var_within, var_estimator


def gelman_rubin(input, sample_dim=0, chain_dim=1):
    """
    Computes R-hat over chains of samples. It is required that
    `input.size(sample_dim) >= 2` and `input.size(chain_dim) >= 2`.

    :param torch.Tensor input: the input tensor.
    :param int sample_dim: the sample dimension.
    :param int chain_dim: the chain dimension.
    :returns torch.Tensor: R-hat of `input`.
    """
    assert input.dim() >= 2
    assert input.size(sample_dim) >= 2
    assert input.size(chain_dim) >= 2
    # change input.shape to 1 x 1 x input.shape
    # then transpose sample_dim with 0, chain_dim with 1
    sample_dim = input.dim() + sample_dim if sample_dim < 0 else sample_dim
    chain_dim = input.dim() + chain_dim if chain_dim < 0 else chain_dim
    assert chain_dim != sample_dim
    input = input.reshape((1, 1) + input.shape)
    input = input.transpose(0, sample_dim + 2).transpose(1, chain_dim + 2)

    var_within, var_estimator = _compute_chain_variance_stats(input)
    rhat = (var_estimator / var_within).sqrt()
    return rhat.squeeze(max(sample_dim, chain_dim)).squeeze(min(sample_dim, chain_dim))


def split_gelman_rubin(input, sample_dim=0, chain_dim=1):
    """
    Computes R-hat over chains of samples. It is required that
    `input.size(sample_dim) >= 4`.

    :param torch.Tensor input: the input tensor.
    :param int sample_dim: the sample dimension.
    :param int chain_dim: the chain dimension.
    :returns torch.Tensor: split R-hat of `input`.
    """
    assert input.dim() >= 2
    assert input.size(sample_dim) >= 4
    # change input.shape to 1 x 1 x input.shape
    # then transpose sample_dim with 0, chain_dim with 1
    sample_dim = input.dim() + sample_dim if sample_dim < 0 else sample_dim
    chain_dim = input.dim() + chain_dim if chain_dim < 0 else chain_dim
    assert chain_dim != sample_dim
    input = input.reshape((1, 1) + input.shape)
    input = input.transpose(0, sample_dim + 2).transpose(1, chain_dim + 2)

    N_half = input.size(0) // 2
    new_input = torch.stack([input[:N_half], input[-N_half:]], dim=1)
    new_input = new_input.reshape((N_half, -1) + input.shape[2:])
    split_rhat = gelman_rubin(new_input)
    return split_rhat.squeeze(max(sample_dim, chain_dim)).squeeze(min(sample_dim, chain_dim))


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


def autocorrelation(input, dim=0):
    """
    Computes the autocorrelation of samples at dimension `dim`.

    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    :param torch.Tensor input: the input tensor.
    :param int dim: the dimension to calculate autocorrelation.
    :returns torch.Tensor: autocorrelation of `input`.
    """
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = _fft_next_good_size(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

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

    # truncate and normalize the result, then transpose back to original shape
    ac = ac[..., :N]
    ac = ac / input.new_tensor(range(N, 0, -1))
    ac = ac / ac[..., :1]
    return ac.transpose(dim, -1)


def autocovariance(input, dim=0):
    """
    Computes the autocovariance of samples at dimension `dim`.

    :param torch.Tensor input: the input tensor.
    :param int dim: the dimension to calculate autocorrelation.
    :returns torch.Tensor: autocorrelation of `input`.
    """
    return autocorrelation(input) * input.var(dim, unbiased=False, keepdim=True)


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


def effective_sample_size(input, chain_dim=0, sample_dim=1):
    """
    Computes effective sample size of input.

    Reference:
    [1] `Introduction to Markov Chain Monte Carlo`,
        Charles J. Geyer
    [2] `Stan Reference Manual version 2.18`,
        Stan Development Team

    :param torch.Tensor input: the input tensor.
    :param int chain_dim: the chain dimension.
    :param int sample_dim: the sample dimension.
    :returns torch.Tensor: effective sample size of `input`.
    """
    assert input.dim() >= 2
    assert input.size(sample_dim) >= 2
    assert input.size(chain_dim) >= 2
    # change input.shape to 1 x 1 x input.shape
    # then transpose sample_dim with 0, chain_dim with 1
    sample_dim = input.dim() + sample_dim if sample_dim < 0 else sample_dim
    chain_dim = input.dim() + chain_dim if chain_dim < 0 else chain_dim
    assert chain_dim != sample_dim
    input = input.reshape((1, 1) + input.shape)
    input = input.transpose(0, sample_dim + 2).transpose(1, chain_dim + 2)

    N, C = input.size(0), input.size(1)
    # find autocovariance for each chain at lag k
    gamma_k_c = autocovariance(input, dim=0)  # N x C x sample_shape

    # find autocorrelation at lag k (from Stan reference)
    var_within, var_estimator = _compute_chain_variance_stats(input)
    rho_k = (var_estimator - var_within + gamma_k_c.mean(dim=1)) / var_estimator
    rho_k[0] = 1  # correlation at lag 0 is always 1

    # initial positive sequence (formula 1.18 in [1]) applied for autocorrelation
    Rho_k = rho_k if N % 2 == 0 else rho_k[:-1]
    Rho_k = Rho_k.reshape((N // 2, 2) + Rho_k.shape[1:]).sum(dim=1)

    # separate the first index
    Rho_init = Rho_k[0]

    if Rho_k.size(0) > 1:
        # Theoretically, Rho_k is positive, but due to noise of correlation computation,
        # Rho_k might not be positive at some point. So we need to truncate (ignore first index).
        Rho_positive = Rho_k[1:].clamp(min=0)

        # Now we make the initial monotone (decreasing) sequence.
        Rho_monotone = _cummin(Rho_positive)

        # Formula 1.19 in [1]
        tau = -1 + 2 * Rho_init + 2 * Rho_monotone.sum(dim=0)
    else:
        tau = -1 + 2 * Rho_init

    n_eff = C * N / tau
    return n_eff.squeeze(max(sample_dim, chain_dim)).squeeze(min(sample_dim, chain_dim))
