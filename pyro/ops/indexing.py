from __future__ import absolute_import, division, print_function

import torch


def broadcasted_getitem(tensor, args):
    """
    Advanced indexing with broadcasting semantics.

    This assumes:

    1. Each arg is either ``Ellipsis``, a ``slice``, or a batched
        :class:`~torch.LongTensor` (i.e. with empty event shape).
    2. If ``args[0] is not Ellipsis`` then ``tensor`` is not
        batched, and its event dim is equal to ``len(args)``.
    3. If ``args[0] is Ellipsis`` then ``tensor`` is batched and
        its event dim is equal to ``len(args[1:])``.

    :param torch.Tensor tensor: A tensor to be indexed.
    :param tuple args: An index, as args to ``__getitem__``.
    :returns: A nonstandard interpetation of ``tensor[args]``.
    :rtype: torch.Tensor
    """
    if not isinstance(args, tuple):
        args = (args,)
    if not args:
        return tensor

    # Compute event dim before and after indexing.
    if args[0] is Ellipsis:
        args = args[1:]
        if not args:
            return tensor
        old_event_dim = len(args)
        args = (slice(None),) * (tensor.dim() - len(args)) + args
    else:
        args = args + (slice(None),) * (tensor.dim() - len(args))
        old_event_dim = len(args)
    assert len(args) == tensor.dim()
    if any(a is Ellipsis for a in args):
        raise NotImplementedError("Non-leading Ellipsis is not supported")

    # In simple cases, standard advanced indexing broadcasts correctly.
    num_batched = (tensor.dim() > old_event_dim)
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.dim():
            num_batched += 1
    if num_batched <= 1:
        return tensor[args]

    # Convert args to use broadcasting semantics.
    new_event_dim = sum(isinstance(a, slice) for a in args[-old_event_dim:])
    new_dim = 0
    args = list(args)
    for i, arg in reversed(list(enumerate(args))):
        if isinstance(arg, slice):
            # Convert slices to torch.arange()s.
            if arg != slice(None):
                raise NotImplementedError("Nontrivial slices are not supported")
            arg = torch.arange(tensor.size(i), dtype=torch.long, device=tensor.device)
            arg = arg.reshape((-1,) + (1,) * new_dim)
            new_dim += 1
        elif isinstance(arg, torch.Tensor) and arg.dim():
            # Reshape nontrivial tensors.
            arg = arg.reshape(arg.shape + (1,) * new_event_dim)
        args[i] = arg
    args = tuple(args)

    return tensor[args]


class broadcasted(object):
    """
    Convenience wrapper around :func:`broadcasted_getitem`.

    The following are equivalent::

        broadcasted(x)[..., i, j, :]
        broadcasted_getitem(x, (Ellipsis, i, j, slice(None)))

    :param torch.Tensor tensor: A tensor to be indexed.
    :return: An object with a special :meth:`__getitem__` method.
    """
    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, args):
        return broadcasted_getitem(self._tensor, args)
