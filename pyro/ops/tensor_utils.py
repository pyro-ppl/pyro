import torch


def block_diag(m):
    """
    Takes a 3-dimensional tensor of shape (B, M, N) and returns a block diagonal tensor
    of shape (B x M, B x N).

    :param torch.Tensor m: 3-dimensional input tensor
    :returns torch.Tensor: a 2-dimensional block diagonal tensor
    """
    assert m.dim() == 3, "Input to block_diag() must be a 3-dimensional tensor"
    batch_size = m.size(0)
    block_shape = m.shape[-2:]
    eye = torch.eye(batch_size).type_as(m).unsqueeze(-2)
    eye = eye.reshape(eye.shape + (1,))
    m = (m.unsqueeze(-2) * eye).reshape(torch.Size(torch.tensor(block_shape) * batch_size))
    return m
