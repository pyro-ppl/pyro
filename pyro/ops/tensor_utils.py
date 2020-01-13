# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch


def block_diag_embed(mat):
    """
    Takes a tensor of shape (..., B, M, N) and returns a block diagonal tensor
    of shape (..., B x M, B x N).

    :param torch.Tensor mat: an input tensor with 3 or more dimensions
    :returns torch.Tensor: a block diagonal tensor with dimension `m.dim() - 1`
    """
    assert mat.dim() > 2, "Input to block_diag() must be of dimension 3 or higher"
    B, M, N = mat.shape[-3:]
    eye = torch.eye(B, dtype=mat.dtype, device=mat.device).reshape(B, 1, B, 1)
    return (mat.unsqueeze(-2) * eye).reshape(mat.shape[:-3] + (B * M, B * N))


def block_diagonal(mat, block_size):
    """
    Takes a block diagonal tensor of shape (..., B x M, B x N) and returns a tensor
    of shape (..., B, M, N).

    :param torch.Tensor mat: an input tensor with 2 or more dimensions
    :param int block_size: the number of blocks B.
    :returns torch.Tensor: a tensor with dimension `mat.dim() + 1`
    """
    B = block_size
    M = mat.size(-2) // B
    N = mat.size(-1) // B
    assert mat.shape[-2:] == (B * M, B * N)
    mat = mat.reshape(mat.shape[:-2] + (B, M, B, N))
    mat = mat.transpose(-2, -3)
    mat = mat.reshape(mat.shape[:-4] + (B * B, M, N))
    return mat[..., ::B + 1, :, :]


def _complex_mul(a, b):
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def convolve(signal, kernel, mode='full'):
    """
    Computes the 1-d convolution of signal by kernel using FFTs.
    The two arguments should have the same rightmost dim, but may otherwise be
    arbitrarily broadcastable.

    :param torch.Tensor signal: A signal to convolve.
    :param torch.Tensor kernel: A convolution kernel.
    :param str mode: One of: 'full', 'valid', 'same'.
    :return: A tensor with broadcasted shape. Letting ``m = signal.size(-1)``
        and ``n = kernel.size(-1)``, the rightmost size of the result will be:
        ``m + n - 1`` if mode is 'full';
        ``max(m, n) - min(m, n) + 1`` if mode is 'valid'; or
        ``max(m, n)`` if mode is 'same'.
    :rtype torch.Tensor:
    """
    m = signal.size(-1)
    n = kernel.size(-1)
    if mode == 'full':
        truncate = m + n - 1
    elif mode == 'valid':
        truncate = max(m, n) - min(m, n) + 1
    elif mode == 'same':
        truncate = max(m, n)
    else:
        raise ValueError('Unknown mode: {}'.format(mode))

    # Compute convolution using fft.
    padded_size = m + n - 1
    # Round up to next power of 2 for cheaper fft.
    fast_ftt_size = 2 ** math.ceil(math.log2(padded_size))
    f_signal = torch.rfft(torch.nn.functional.pad(signal, (0, fast_ftt_size - m)), 1, onesided=False)
    f_kernel = torch.rfft(torch.nn.functional.pad(kernel, (0, fast_ftt_size - n)), 1, onesided=False)
    f_result = _complex_mul(f_signal, f_kernel)
    result = torch.irfft(f_result, 1, onesided=False)

    start_idx = (padded_size - truncate) // 2
    return result[..., start_idx: start_idx + truncate]


def repeated_matmul(M, n):
    """
    Takes a batch of matrices `M` as input and returns the stacked result of doing the
    `n`-many matrix multiplications :math:`M`, :math:`M^2`, ..., :math:`M^n`.
    Parallel cost is logarithmic in `n`.

    :param torch.Tensor M: A batch of square tensors of shape (..., N, N).
    :param int n: The order of the largest product :math:`M^n`
    :returns torch.Tensor: A batch of square tensors of shape (n, ..., N, N)
    """
    assert M.size(-1) == M.size(-2), "Input tensors must satisfy M.size(-1) == M.size(-2)."
    assert n > 0, "argument n to parallel_scan_repeated_matmul must be 1 or larger"

    doubling_rounds = 0 if n <= 2 else math.ceil(math.log(n, 2)) - 1

    if n == 1:
        return M.unsqueeze(0)

    result = torch.stack([M, torch.matmul(M, M)])

    for i in range(doubling_rounds):
        doubled = torch.matmul(result[-1].unsqueeze(0), result)
        result = torch.stack([result, doubled]).reshape(-1, *result.shape[1:])

    return result[0:n]


def _real_of_complex_mul(a, b):
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return ar * br - ai * bi


def dct(x):
    """
    Discrete cosine transform of type II, scaled to be orthonormal.

    This is the inverse of :func:`idct_ii` , and is equivalent to
    :func:`scipy.fftpack.dct` with ``norm="ortho"``.

    :param Tensor x:
    :rtype: Tensor
    """
    # Ref: http://fourier.eng.hmc.edu/e161/lectures/dct/node2.html
    N = x.size(-1)
    # Step 1
    y = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    # Step 2
    Y = torch.rfft(y, 1, onesided=False)
    # Step 3
    coef_real = torch.cos(torch.linspace(0, 0.5 * math.pi, N + 1, dtype=x.dtype, device=x.device))
    coef = torch.stack([coef_real[:-1], -coef_real[1:].flip(-1)], dim=-1)
    X = _real_of_complex_mul(coef, Y)
    # orthogonalize
    scale = torch.cat([x.new_tensor([math.sqrt(N)]), x.new_full((N - 1,), math.sqrt(0.5 * N))])
    return X / scale


def idct(x):
    """
    Inverse discrete cosine transform of type II, scaled to be orthonormal.

    This is the inverse of :func:`dct_ii` , and is equivalent to
    :func:`scipy.fftpack.idct` with ``norm="ortho"``.

    :param Tensor x: The input signal
    :rtype: Tensor
    """
    N = x.size(-1)
    scale = torch.cat([x.new_tensor([math.sqrt(N)]), x.new_full((N - 1,), math.sqrt(0.5 * N))])
    x = x * scale
    # Step 1, solve X = cos(k) * Yr + sin(k) * Yi
    # We know that Y[1:] is conjugate to Y[:0:-1], hence
    # X[:0:-1] = sin(k) * Yr[1:] + cos(k) * Yi[1:]
    # So Yr[1:] = cos(k) * X[1:] + sin(k) * X[:0:-1]
    # and Yi[1:] = sin(k) * X[1:] - cos(k) * X[:0:-1]
    # In addition, Yi[0] = 0, Yr[0] = X[0]
    # In other words, Y = complex_mul(e^ik, X - i[0, X[:0:-1]])
    xi = torch.nn.functional.pad(-x[..., 1:], (0, 1)).flip(-1)
    X = torch.stack([x, xi], dim=-1)
    coef_real = torch.cos(torch.linspace(0, 0.5 * math.pi, N + 1))
    coef = torch.stack([coef_real[:-1], coef_real[1:].flip(-1)], dim=-1)
    half_size = N // 2 + 1
    Y = _complex_mul(coef[..., :half_size, :], X[..., :half_size, :])
    # Step 2
    y = torch.irfft(Y, 1, onesided=True, signal_sizes=(N,))
    # Step 3
    return torch.stack([y, y.flip(-1)], axis=-1).reshape(x.shape[:-1] + (-1,))[..., :N]
