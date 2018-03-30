from __future__ import absolute_import, division, print_function

import numbers

import torch
import torch.distributions as torch_dist
import torch.nn.functional as F


_VALIDATION_ENABLED = False


def copy_docs_from(source_class, full_text=False):
    """
    Decorator to copy class and method docs from source to destin class.
    """

    def decorator(destin_class):
        # This works only in python 3.3+:
        # if not destin_class.__doc__:
        #     destin_class.__doc__ = source_class.__doc__
        for name in dir(destin_class):
            if name.startswith('_'):
                continue
            destin_attr = getattr(destin_class, name)
            destin_attr = getattr(destin_attr, '__func__', destin_attr)
            source_attr = getattr(source_class, name, None)
            source_doc = getattr(source_attr, '__doc__', None)
            if source_doc and not getattr(destin_attr, '__doc__', None):
                if full_text or source_doc.startswith('See '):
                    destin_doc = source_doc
                else:
                    destin_doc = 'See :meth:`{}.{}.{}`'.format(
                        source_class.__module__, source_class.__name__, name)
                if isinstance(destin_attr, property):
                    # Set docs for object properties.
                    # Since __doc__ is read-only, we need to reset the property
                    # with the updated doc.
                    updated_property = property(destin_attr.fget,
                                                destin_attr.fset,
                                                destin_attr.fdel,
                                                destin_doc)
                    setattr(destin_class, name, updated_property)
                else:
                    destin_attr.__doc__ = destin_doc
        return destin_class

    return decorator


def is_identically_zero(x):
    """
    Check if argument is exactly the number zero. True for the number zero;
    false for other numbers; false for :class:`~torch.Tensor`s.
    """
    return isinstance(x, numbers.Number) and x == 0


def is_identically_one(x):
    """
    Check if argument is exactly the number one. True for the number one;
    false for other numbers; false for :class:`~torch.Tensor`s.
    """
    return isinstance(x, numbers.Number) and x == 1


def broadcast_shape(*shapes, **kwargs):
    """
    Similar to ``np.broadcast()`` but for shapes.
    Equivalent to ``np.broadcast(*map(np.empty, shapes)).shape``.

    :param tuple shapes: shapes of tensors.
    :param bool strict: whether to use extend-but-not-resize broadcasting.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    strict = kwargs.pop('strict', False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError('shape mismatch: objects cannot be broadcast to a single shape: {}'.format(
                    ' vs '.join(map(str, shapes))))
    return tuple(reversed(reversed_shape))


def sum_rightmost(value, dim):
    """
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    If ``dim`` is 0, no dimensions are summed out.
    If ``dim`` is ``float('inf')``, then all dimensions are summed out.
    If ``dim`` is 1, the rightmost 1 dimension is summed out.
    If ``dim`` is 2, the rightmost two dimensions are summed out.
    If ``dim`` is -1, all but the leftmost 1 dimension is summed out.
    If ``dim`` is -2, all but the leftmost 2 dimensions are summed out.
    etc.

    :param torch.Tensor value: A tensor of ``.dim()`` at least ``dim``.
    :param int dim: The number of rightmost dims to sum out.
    """
    if isinstance(value, numbers.Number):
        return value
    if dim < 0:
        dim += value.dim()
    if dim == 0:
        return value
    if dim >= value.dim():
        return value.sum()
    return value.contiguous().view(value.shape[:-dim] + (-1,)).sum(-1)


def sum_leftmost(value, dim):
    """
    Sum out ``dim`` many leftmost dimensions of a given tensor.

    If ``dim`` is 0, no dimensions are summed out.
    If ``dim`` is ``float('inf')``, then all dimensions are summed out.
    If ``dim`` is 1, the leftmost 1 dimension is summed out.
    If ``dim`` is 2, the leftmost two dimensions are summed out.
    If ``dim`` is -1, all but the rightmost 1 dimension is summed out.
    If ``dim`` is -2, all but the rightmost 2 dimensions are summed out.
    etc.

    Example::

        x = torch.ones(2, 3, 4)
        assert sum_leftmost(x, 1).shape == (3, 4)
        assert sum_leftmost(x, -1).shape == (4,)

    :param torch.Tensor value: A tensor
    :param int dim: Specifies the number of dims to sum out
    """
    if isinstance(value, numbers.Number):
        return value
    if dim < 0:
        dim += value.dim()
    if dim == 0:
        return value
    if dim >= value.dim():
        return value.sum()
    return value.contiguous().view(-1, *value.shape[dim:]).sum(0)


def scale_tensor(tensor, scale):
    """
    Safely scale a tensor without increasing its ``.size()``.
    This avoids NANs by assuming ``inf * 0 = 0 * inf = 0``.
    """
    if isinstance(tensor, numbers.Number):
        if isinstance(scale, numbers.Number):
            return tensor * scale
        elif tensor == 0:
            return torch.zeros_like(scale)
        elif tensor == 1:
            return scale
        else:
            return scale
    if isinstance(scale, numbers.Number):
        if scale == 0:
            return torch.zeros_like(tensor)
        elif scale == 1:
            return tensor
        else:
            return tensor * scale
    result = tensor * scale
    result[(scale == 0).expand_as(result)] = 0  # avoid NANs
    if result.shape != tensor.shape:
        raise ValueError("Broadcasting error: scale is incompatible with tensor: "
                         "{} vs {}".format(scale.shape, tensor.shape))
    return result


def torch_eye(n, m=None, out=None):
    """
    Like :func:`torch.eye`, but works with cuda tensors.
    """
    if m is None:
        m = n
    try:
        return torch.eye(n, m, out=out)
    except TypeError:
        # Only catch errors due to torch.eye() not being available for cuda tensors.
        module = torch.Tensor.__module__ if out is None else type(out).__module__
        if module != 'torch.cuda':
            raise
    Tensor = getattr(torch, torch.Tensor.__name__)
    cpu_out = Tensor(n, m)
    cuda_out = torch.eye(m, n, out=cpu_out).cuda()
    return cuda_out if out is None else out.copy_(cuda_out)


def torch_multinomial(input, num_samples, replacement=False):
    """
    Like :func:`torch.multinomial` but works with cuda tensors.
    Does not support keyword argument `out`.
    """
    if input.is_cuda:
        return torch.multinomial(input.cpu(), num_samples, replacement).cuda(input.get_device())
    else:
        return torch.multinomial(input, num_samples, replacement)


def torch_sign(value):
    """
    Like :func:`torch.sign`` but also works for numbers.
    """
    if isinstance(value, numbers.Number):
        return (value > 0) - (value < 0)
    return torch.sign(value)


def softmax(x, dim=-1):
    """
    TODO: change to use the default pyTorch implementation when available
    Source: https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
    :param x: tensor
    :param dim: Dimension to apply the softmax function to. The elements of the tensor in this
        dimension must sum to 1.
    :return: tensor having the same dimension as `x` rescaled along dim
    """
    input_size = x.size()

    trans_input = x.transpose(dim, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d, 1)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(dim, len(input_size) - 1)


def _get_clamping_buffer(tensor):
    clamp_eps = 1e-6
    if isinstance(tensor, (torch.DoubleTensor, torch.cuda.DoubleTensor)):
        clamp_eps = 1e-15
    return clamp_eps


def get_probs_and_logits(probs=None, logits=None, is_multidimensional=True):
    """
    Convert probability values to logits, or vice-versa. Either ``probs`` or
    ``logits`` should be specified, but not both.

    :param probs: tensor of probabilities. Should be in the interval *[0, 1]*.
        If, ``is_multidimensional = True``, then must be normalized along
        axis -1.
    :param logits: tensor of logit values.  For the multidimensional case,
        the values, when exponentiated along the last dimension, must sum
        to 1.
    :param is_multidimensional: determines the computation of probs from logits,
        and vice-versa. For the multi-dimensional case, logit values are
        assumed to be log probabilities, whereas for the uni-dimensional case,
        it specifically refers to log odds.
    :return: tuple containing raw probabilities and logits as tensors.
    """
    assert (probs is None) != (logits is None)
    if probs is not None:
        eps = _get_clamping_buffer(probs)
        ps_clamped = probs.clamp(min=eps, max=1 - eps)
    if is_multidimensional:
        if probs is None:
            probs = softmax(logits, -1)
        else:
            logits = torch.log(ps_clamped)
    else:
        if probs is None:
            probs = F.sigmoid(logits)
        else:
            logits = torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return probs, logits


def get_clamped_probs(probs=None, logits=None, is_multidimensional=True):
    """
    Clamp probabilities, given probability values or logits. Either ``probs`` or
    ``logits`` should be specified, but not both.

    :param probs: tensor of probabilities. Should be in the interval *[0, 1]*.
        If, ``is_multidimensional = True``, then must be normalized along
        axis -1.
    :param logits: tensor of logit values.  For the multidimensional case,
        the values, when exponentiated along the last dimension, must sum
        to 1.
    :param is_multidimensional: determines the computation of probs from logits,
        and vice-versa. For the multi-dimensional case, logit values are
        assumed to be log probabilities, whereas for the uni-dimensional case,
        it specifically refers to log odds.
    :return: clamped probabilities.
    """
    if (probs is None) == (logits is None):
        raise ValueError("Got probs={}, logits={}. Either `probs` or `logits` must be specified, "
                         "but not both.".format(probs, logits))
    if probs is None:
        probs = softmax(logits, -1) if is_multidimensional else F.sigmoid(logits)
    eps = _get_clamping_buffer(probs)
    probs = probs.clamp(min=eps, max=1 - eps)
    if is_multidimensional:
        probs /= probs.sum(-1, True)
    return probs


def matrix_triangular_solve_compat(b, A, upper=True):
    """
    Computes the solution to the linear equation AX = b,
    where A is a triangular matrix.

    :param b: A 1D or 2D tensor of size N or N x C.
    :param A: A 2D tensor of size N X N.
    :param upper: A flag if A is a upper triangular matrix or not.
    """
    if A.requires_grad or A.is_cuda:
        return A.inverse().matmul(b)
    else:
        return b.trtrs(A, upper=upper)[0].view(b.size())


def enable_validation(is_validate):
    global _VALIDATION_ENABLED
    _VALIDATION_ENABLED = is_validate
    torch_dist.Distribution.set_default_validate_args(is_validate)


def is_validation_enabled():
    return _VALIDATION_ENABLED
