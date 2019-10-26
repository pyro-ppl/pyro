import math

import torch


def block_diag(m):
    """
    Takes a tensor of shape (..., B, M, N) and returns a block diagonal tensor
    of shape (..., B x M, B x N).

    :param torch.Tensor m: an input tensor with 3 or more dimensions
    :returns torch.Tensor: a block diagonal tensor with dimension `m.dim() - 1`
    """
    assert m.dim() > 2, "Input to block_diag() must be of dimension 3 or higher"
    B, M, N = m.shape[-3:]
    eye = torch.eye(B, dtype=m.dtype, device=m.device).reshape(B, 1, B, 1)
    return (m.unsqueeze(-2) * eye).reshape(m.shape[:-3] + (B * M, B * N))


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
