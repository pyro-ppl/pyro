from __future__ import absolute_import, division, print_function

from numbers import Number

import torch


def _patch(target):
    parts = target.split('.')
    assert parts[0] == 'torch'
    module = torch
    for part in parts[1:-1]:
        module = getattr(module, part)
    name = parts[-1]
    old_fn = getattr(module, name)
    old_fn = getattr(old_fn, '_pyro_unpatched', old_fn)  # ensure patching is idempotent

    def decorator(new_fn):
        new_fn.__name__ = name
        new_fn._pyro_unpatched = old_fn
        setattr(module, name, new_fn)
        return new_fn

    return decorator


@_patch('torch._standard_gamma')
def _torch_standard_gamma(concentration):
    unpatched_fn = _torch_standard_gamma._pyro_unpatched
    if concentration.is_cuda:
        return unpatched_fn(concentration.cpu()).cuda(concentration.get_device())
    return unpatched_fn(concentration)


@_patch('torch.distributions.gamma._standard_gamma')
def _standard_gamma(concentration):
    if concentration.is_cuda:
        return concentration.cpu()._standard_gamma().cuda(concentration.get_device())
    return concentration._standard_gamma()


@_patch('torch._dirichlet_grad')
def _torch_dirichlet_grad(x, concentration, total):
    unpatched_fn = _torch_dirichlet_grad._pyro_unpatched
    if x.is_cuda:
        return unpatched_fn(x.cpu(), concentration.cpu(), total.cpu()).cuda(x.get_device())
    return unpatched_fn(x, concentration, total)


# This version of broadcast_all() is compatible with early versions of the PyTorch jit,
# since it avoids torch._C._infer_size(). However it is more expensive since it infers
# size by summing the tensors. It is mainly useful for working around one jit limitation
# to discovering additional jit limitations.
#
# To temporarily apply this patch, uncomment one or more of the decorators:
#
# @_patch('torch.distributions.beta.broadcast_all')
# @_patch('torch.distributions.dirichlet.broadcast_all')
# @_patch('torch.distributions.normal.broadcast_all')
# @_patch('torch.distributions.utils.broadcast_all')
def _broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per the `broadcasting rules
        <http://pytorch.org/docs/master/notes/broadcasting.html>`_
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to Tensors having size
        `(1,)`.

    Args:
        values (list of `numbers.Number` or `torch.*Tensor`)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` or
            `torch.*Tensor` instance
    """
    values = list(values)
    scalar_idxs = [i for i in range(len(values)) if isinstance(values[i], Number)]
    tensor_idxs = [i for i in range(len(values)) if values[i].__class__.__name__ == 'Tensor']
    if len(scalar_idxs) + len(tensor_idxs) != len(values):
        raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
    if tensor_idxs:
        broadcast_shape = sum(values).size()  # expensive alternative to torch._C._infer_size()
        for idx in tensor_idxs:
            values[idx] = values[idx].expand(broadcast_shape)
        template = values[tensor_idxs[0]]
        for idx in scalar_idxs:
            values[idx] = template.new(template.size()).fill_(values[idx])
    else:
        for idx in scalar_idxs:
            values[idx] = torch.tensor(float(values[idx]))
    return values


__all__ = []
