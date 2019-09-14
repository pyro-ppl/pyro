import torch


def fn(x):
    return x[..., 0]


traced_fn = torch.jit.trace(fn, (torch.zeros(1),))


print(traced_fn(torch.ones(1)))
