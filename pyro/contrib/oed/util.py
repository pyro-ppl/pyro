import torch

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
