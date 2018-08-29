import torch
from torch.distributions.multivariate_normal import _batch_inverse


def get_indices(labels, sizes=None, tensors=None):
    indices = []
    start = 0
    for label in labels:
        if sizes is not None:
            end = start+sizes[label][0]
        else:
            end = start+tensors[label].shape[0]
        indices.extend(range(start, end))
        start = end
    return torch.tensor(indices)


def rmm(A, B):
    """Shorthand for `matmul`."""
    return torch.matmul(A, B)


def rmv(A, b):
    """Tensorized matrix vector multiplication of rightmost dimensions."""
    return torch.matmul(A, b.unsqueeze(-1)).squeeze(-1)


def rvv(a, b):
    """Tensorized vector vector multiplication of rightmost dimensions."""
    return torch.matmul(a.unsqueeze(-2), b.unsqueeze(-1)).squeeze(-2).squeeze(-1)


def lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    return A.expand(tuple(dimensions) + A.shape)


def rexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on right."""
    return A.expand(A.shape + tuple(dimensions))


def rinverse(M):
    """Matrix inversion of rightmost dimensions (batched).
    Accelerated for 1x1 and 2x2 matrices.
    """
    assert M.shape[-1] == M.shape[-2]
    if M.shape[-1] == 1:
        return 1./M
    elif M.shape[-1] == 2:
        det = M[..., 0, 0]*M[..., 1, 1] - M[..., 1, 0]*M[..., 0, 1]
        inv = torch.zeros(M.shape)
        inv[..., 0, 0] = M[..., 1, 1]
        inv[..., 1, 1] = M[..., 0, 0]
        inv[..., 0, 1] = -M[..., 0, 1]
        inv[..., 1, 0] = -M[..., 1, 0]
        inv = inv/det.unsqueeze(-1).unsqueeze(-1)
        return inv
    else:
        return _batch_inverse(M)


def rdiag(v):
    """Converts the rightmost dimension to a diagonal matrix."""
    return rexpand(v, v.shape[-1])*torch.eye(v.shape[-1])


def rtril(M, diagonal=0):
    """Takes the lower-triangular of the rightmost 2 dimensions."""
    return M*torch.tril(torch.ones(M.shape[-2], M.shape[-1]), diagonal=diagonal)
