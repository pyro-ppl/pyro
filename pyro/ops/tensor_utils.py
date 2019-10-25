import math

import torch


def block_diag(m):
    """
    Takes a 3-dimensional tensor of shape (B, M, N) and returns a block diagonal tensor
    of shape (B x M, B x N).

    :param torch.Tensor m: 3-dimensional input tensor
    :returns torch.Tensor: a 2-dimensional block diagonal tensor
    """
    assert m.dim() == 3, "Input to block_diag() must be a 3-dimensional tensor"
    B, M, N = m.shape
    eye = torch.eye(B, dtype=m.dtype, device=m.device).reshape(B, 1, B, 1)
    return (m.unsqueeze(-2) * eye).reshape(B * M, B * N)


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

    result = torch.stack([M, torch.matmul(M, M)])

    for i in range(doubling_rounds):
        doubled = torch.matmul(result[-1].unsqueeze(0), result)
        result = torch.stack([result, doubled]).reshape(-1, *result.shape[1:])

    return result[0:n]
