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
    # FIXME should b be conjugated?
    # bi = -bi  # Conjugates b.
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def conv1d_fft(signal, kernel):
    """
    Computes the 1-d convolution of signal by kernel using FFTs.
    The two arguments should have the same rightmost dim, but may otherwise be
    arbitrarily broadcastable.

    :param torch.Tensor signal: A signal to convolve.
    :param torch.Tensor kernel: A convolution kernel.
    :return: A tensor of shape
        ``broadcast_shape(convolve, kernel.shape)``
    :rtype torch.Tensor:
    """
    assert signal.size(-1) == kernel.size(-1)
    f_signal = torch.rfft(torch.cat([signal, torch.zeros_like(signal)], dim=-1), 1)
    f_kernel = torch.rfft(torch.cat([kernel, torch.zeros_like(kernel)], dim=-1), 1)
    f_result = _complex_mul(f_signal, f_kernel)
    return torch.irfft(f_result, 1)[..., :signal.size(-1)]
