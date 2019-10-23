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
