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
