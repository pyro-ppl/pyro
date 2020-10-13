# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import ctypes
import functools
import numbers
import weakref
from contextlib import contextmanager

import torch
import torch.distributions as torch_dist
from torch import logsumexp
from torch.distributions.utils import broadcast_all

from pyro.util import ignore_jit_warnings

_VALIDATION_ENABLED = False

log_sum_exp = logsumexp  # DEPRECATED


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


# TODO replace with weakref.WeakMethod?
def weakmethod(fn):
    """
    Decorator to enforce weak binding of a method, so as to avoid reference
    cycles when passing a bound method as an argument to other functions.

    In the following example, functional behavior is the same with and without
    the ``@weakmethod`` decorator, but decoration avoids a reference cycle::

        class Foo:
            def __init__(self):
                self.callback = self._callback
            @weakmethod
            def _callback(self, result):
                print(result)
    """
    def weak_fn(weakself, *args, **kwargs):
        self = weakself()
        if self is None:
            raise AttributeError("self was garbage collected when calling self.{}"
                                 .format(fn.__name__))
        return fn(self, *args, **kwargs)

    @property
    def weak_binder(self):
        weakself = weakref.ref(self)
        return functools.partial(weak_fn, weakself)

    @weak_binder.setter
    def weak_binder(self, new):
        if not (isinstance(new, functools.partial) and new.func is weak_fn and
                len(new.args) == 1 and new.args[0] is weakref.ref(self)):
            raise AttributeError("cannot overwrite weakmethod {}".format(fn.__name__))

    return weak_binder


# This helper intervenes in copy.deepcopy's use of a memo dict to
# use .detatch() to copy tensors.
class _DetachMemo(dict):
    def get(self, key, default=None):
        result = super().get(key, default)

        if result is default:
            # Assume key is the id of another object, and look up that object.
            old = ctypes.cast(key, ctypes.py_object).value
            if isinstance(old, torch.Tensor):
                self[key] = result = old.detach()

        return result


def detach(obj):
    """
    Create a deep copy of an object, detaching all :class:`torch.Tensor` s in
    the object. No tensor data is actually copied.

    :param obj: Any python object.
    :returns: A deep copy of ``obj`` with all :class:`torch.Tensor` s detached.
    """
    memo = _DetachMemo()
    return copy.deepcopy(obj, memo)


# Import a lazy jit.script decorator only if available.
torch_jit_script_if_tracing = getattr(
    torch.jit,
    "script_if_tracing",
    getattr(
        torch.jit,
        "_script_if_tracing",
        torch.jit.script
    ),
)


def is_identically_zero(x):
    """
    Check if argument is exactly the number zero. True for the number zero;
    false for other numbers; false for :class:`~torch.Tensor`s.
    """
    if isinstance(x, numbers.Number):
        return x == 0
    if not torch._C._get_tracing_state():
        if isinstance(x, torch.Tensor) and x.dtype == torch.int64 and not x.shape:
            return x.item() == 0
    return False


def is_identically_one(x):
    """
    Check if argument is exactly the number one. True for the number one;
    false for other numbers; false for :class:`~torch.Tensor`s.
    """
    if isinstance(x, numbers.Number):
        return x == 1
    if not torch._C._get_tracing_state():
        if isinstance(x, torch.Tensor) and x.dtype == torch.int64 and not x.shape:
            return x.item() == 1
    return False


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


def gather(value, index, dim):
    """
    Broadcasted gather of indexed values along a named dim.
    """
    value, index = broadcast_all(value, index)
    with ignore_jit_warnings():
        zero = torch.zeros(1, dtype=torch.long, device=index.device)
    index = index.index_select(dim, zero)
    return value.gather(dim, index)


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
    return value.reshape(value.shape[:-dim] + (-1,)).sum(-1)


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
    return value.reshape(-1, *value.shape[dim:]).sum(0)


def scale_and_mask(tensor, scale=1.0, mask=None):
    """
    Scale and mask a tensor, broadcasting and avoiding unnecessary ops.

    :param tensor: an input tensor or zero
    :type tensor: torch.Tensor or the number zero
    :param scale: a positive scale
    :type scale: torch.Tensor or number
    :param mask: an optional masking tensor
    :type mask: torch.BoolTensor or None
    """
    if is_identically_zero(tensor) or (mask is None and is_identically_one(scale)):
        return tensor
    if mask is None:
        return tensor * scale
    return torch.where(mask, tensor * scale, tensor.new_zeros(()))


def scalar_like(prototype, fill_value):
    return torch.tensor(fill_value, dtype=prototype.dtype, device=prototype.device)


# work around lack of jit support for torch.eye(..., out=value)
def eye_like(value, m, n=None):
    if n is None:
        n = m
    eye = torch.zeros(m, n, dtype=value.dtype, device=value.device)
    eye.view(-1)[:min(m, n) * n:n + 1] = 1
    return eye


def enable_validation(is_validate):
    global _VALIDATION_ENABLED
    _VALIDATION_ENABLED = is_validate
    torch_dist.Distribution.set_default_validate_args(is_validate)


def is_validation_enabled():
    return _VALIDATION_ENABLED


@contextmanager
def validation_enabled(is_validate=True):
    distribution_validation_status = is_validation_enabled()
    try:
        enable_validation(is_validate)
        yield
    finally:
        enable_validation(distribution_validation_status)
