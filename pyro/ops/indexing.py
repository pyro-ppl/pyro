from __future__ import absolute_import, division, print_function

import torch


def _is_batched(arg):
    return isinstance(arg, torch.Tensor) and arg.dim()


def _is_scalar(arg):
    return isinstance(arg, int) or (isinstance(arg, torch.Tensor) and not arg.dim())


def _generalized_slice(tensor, args, old_event_dim):
    """
    This generalizes Python slicing to include torch.aranges that have been
    unsqueezed on the right to accomplish a permutation. These special aranges
    are indicated by torch.Tensors with an attribute ._pyro_generalized_slice.
    """
    # Construct a permutation.
    new_event_dim = old_event_dim - sum(1 for a in args if not isinstance(a, slice))
    permutation_map = {}
    source = -1
    destin = -1
    for arg in reversed(args):
        if _is_scalar(arg):
            continue
        elif isinstance(arg, slice):
            permutation_map[source] = destin
            destin -= 1
        else:
            permutation_map[source] = -arg.dim() - new_event_dim
        source -= 1
    if len(set(permutation_map.values())) < len(permutation_map):
        return None  # On collision, fall back to aranges.
    new_dim = -min(permutation_map.values())
    sources = set(range(-new_dim, 0))
    permutation = [None] * new_dim
    for source, destin in permutation_map.items():
        permutation[destin] = source
        sources.remove(source)
    sources = sorted(sources)
    for destin, source in enumerate(permutation):
        if source is None:
            permutation[destin] = sources.pop()
    assert not sources

    # Slice and permute the tensor.
    tensor = tensor[tuple(getattr(a, "_pyro_generalized_slice", a) for a in args)]
    missing = new_dim - tensor.dim()
    assert missing >= 0
    if missing:
        tensor = tensor.reshape((1,) * missing + tensor.size())
    return tensor.permute(permutation)


def vindex(tensor, args):
    """
    Vectorized advanced indexing with broadcasting semantics.

    See also the convenience wrapper :class:`Vindex`.

    This is useful for writing indexing code that is compatible with batching
    and enumeration, especially for selecting mixture components with discrete
    random variables.

    For example suppose ``x`` is a parameter with ``x.dim() == 3`` and we wish
    to generalize the expression ``x[i, :, j]`` from integer ``i,j`` to tensors
    ``i,j`` with batch dims and enum dims (but no event dims). Then we can
    write the generalize version using :class:`Vindex` ::

        xij = Vindex(x)[i, :, j]

        batch_shape = broadcast_shape(i.shape, j.shape)
        event_shape = (x.size(1),)
        assert xij.shape == batch_shape + event_shape

    To handle the case when ``x`` may also contain batch dimensions (e.g. if
    ``x`` was sampled in a plated context as when using vectorized particles),
    :func:`vindex` uses the special convention that ``Ellipsis`` denotes batch
    dimensions (hence ``...`` can appear only on the left, never in the middle
    or in the right). Suppose ``x`` has event dim 3. Then we can write::

        old_batch_shape = x.shape[:-3]
        old_event_shape = x.shape[-3:]

        xij = Vindex(x)[..., i, :, j]   # The ... denotes unknown batch shape.

        new_batch_shape = broadcast_shape(old_batch_shape, i.shape, j.shape)
        new_event_shape = (x.size(1),)
        assert xij.shape = new_batch_shape + new_event_shape

    Note that this special handling of ``Ellipsis`` differs from the NEP [1].

    Formally, this function assumes:

    1.  Each arg is either ``Ellipsis``, ``slice(None)``, an integer, or a
        batched ``torch.LongTensor`` (i.e. with empty event shape). This
        function does not support Nontrivial slices or ``torch.ByteTensor``
        masks. ``Ellipsis`` can only appear on the left as ``args[0]``.
    2.  If ``args[0] is not Ellipsis`` then ``tensor`` is not
        batched, and its event dim is equal to ``len(args)``.
    3.  If ``args[0] is Ellipsis`` then ``tensor`` is batched and
        its event dim is equal to ``len(args[1:])``. Dims of ``tensor``
        to the left of the event dims are considered batch dims and will be
        broadcasted with dims of tensor args.

    Note that if none of the args is a tensor with ``.dim() > 0``, then this
    function behaves like standard indexing::

        if not any(isinstance(a, torch.Tensor) and a.dim() for a in args):
            assert Vindex(x)[args] == x[args]

    **References**

    [1] https://www.numpy.org/neps/nep-0021-advanced-indexing.html
        introduces ``vindex`` as a helper for vectorized indexing.
        The Pyro implementation is similar to the proposed notation
        ``x.vindex[]`` except for slightly different handling of ``Ellipsis``.

    :param torch.Tensor tensor: A tensor to be indexed.
    :param tuple args: An index, as args to ``__getitem__``.
    :returns: A nonstandard interpetation of ``tensor[args]``.
    :rtype: torch.Tensor
    """
    if not isinstance(args, tuple):
        if not hasattr(args, "_pyro_generalized_slice"):
            return tensor[args]
        args = (args,)
    if not any(_is_batched(a) for a in args):
        return tensor[args]

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

    # If generalized slicing is possible, slice-and-permute to avoid touching data.
    if all(hasattr(a, "_pyro_generalized_slice") or not _is_batched(a) for a in args):
        result = _generalized_slice(tensor, args, old_event_dim)
        if result is not None:
            return result

    # In simple cases, standard advanced indexing broadcasts correctly.
    is_standard = True
    if tensor.dim() > old_event_dim and _is_batched(args[0]):
        is_standard = False
    elif any(_is_batched(a) for a in args[1:]):
        is_standard = False
    if is_standard:
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
        elif _is_batched(arg):
            # Reshape nontrivial tensors.
            arg = arg.reshape(arg.shape + (1,) * new_event_dim)
        args[i] = arg
    args = tuple(args)

    return tensor[args]


class Vindex(object):
    """
    Convenience wrapper around :func:`vindex`.

    The following are equivalent::

        Vindex(x)[..., i, j, :]
        vindex(x, (Ellipsis, i, j, slice(None)))

    :param torch.Tensor tensor: A tensor to be indexed.
    :return: An object with a special :meth:`__getitem__` method.
    """
    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, args):
        return vindex(self._tensor, args)
