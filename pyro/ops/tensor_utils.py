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
    target_shape = m.shape[:-3] + (B * M, B * N)
    return (m.unsqueeze(-2) * eye).reshape(target_shape)


def _double(M):
    """
    Internal helper function for parallel_scan_repeated_matmul
    """
    eye = torch.eye(M.size(-1), dtype=M.dtype, device=M.device)
    eye = eye.expand(M[-1, ...].shape)
    doubler = torch.stack([eye, M[-1, ...]]).unsqueeze(1)
    doubled = torch.matmul(doubler, M).reshape(-1, *M.shape[1:])
    return doubled


def parallel_scan_repeated_matmul(M, n):
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

    doubling_rounds = math.ceil(math.log(n, 2))

    Msq = torch.matmul(M, M)
    result = torch.stack([M, Msq])

    for i in range(doubling_rounds):
        result = _double(result)

    return result[0:n, ...]
