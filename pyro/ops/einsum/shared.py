from __future__ import absolute_import, division, print_function

import contextlib
import numbers
from collections import OrderedDict

import opt_einsum
from opt_einsum.backends.dispatch import get_func

_SHARING_STACK = []
_CURRENT_BACKEND = []
_PATH_CACHE = {}
LAST_CACHE_SIZE = [0]  # for profiling


@contextlib.contextmanager
def shared_intermediates(cache=None):
    """
    Context in which :func:`~pyro.ops.einsum.contract` intermediate results are shared.
    Note that intermediate computations will not be garbage collected until
    1. this context exits, and
    2. the yielded cache is garbage collected (if it was captured).

    :param dict cache: an optional dict
    :yields: a cache dict suitable for saving and reusing as an input
    """
    if cache is None:
        cache = {}
    _SHARING_STACK.append(cache)
    yield cache
    LAST_CACHE_SIZE[0] = len(cache)
    _SHARING_STACK.pop()


def contract(equation, *operands, **kwargs):
    """
    Like :func:`opt_einsum.contract` but works with
    :func:`~pyro.ops.einsum.shared_intermediates` contexts.

    :param bool cache_path: whether to cache the contraction path.
        Defaults to True.
    """
    backend = kwargs.pop('backend', 'numpy')
    cache_path = kwargs.pop('cache_path', True)

    # special handling under shared_intermediates()
    if _SHARING_STACK and not _CURRENT_BACKEND:
        operands_ = [_shared(t) for t in operands]
        _CURRENT_BACKEND.append(backend)
        backend = 'pyro.ops.einsum.shared'
        try:
            result_ = contract(equation, *operands_, backend=backend, **kwargs)
        finally:
            _CURRENT_BACKEND.pop()
        return result_._value

    if backend == 'pyro.ops.einsum.shared' and not _CURRENT_BACKEND:
        raise ValueError('shared backend is available only via shared_intermediates')

    if not cache_path:
        return opt_einsum.contract(equation, *operands, backend=backend, **kwargs)

    # memoize the contraction path
    out = kwargs.pop('out', None)
    kwargs_key = tuple(kwargs.items())
    shapes = tuple(tuple(t.shape) for t in operands)
    key = equation, shapes, kwargs_key
    if key in _PATH_CACHE:
        expr = _PATH_CACHE[key]
    else:
        expr = opt_einsum.contract_expression(equation, *shapes, **kwargs)
        _PATH_CACHE[key] = expr
    return expr(*operands, backend=backend, out=out)


def _alpha_canonicalize(equation):
    """
    Alpha convert in an order-independent canonical way.
    """
    rename = OrderedDict()
    for name in equation:
        if name in ',->':
            continue
        if name not in rename:
            rename[name] = opt_einsum.get_symbol(len(rename))
    return ''.join(rename.get(x, x) for x in equation)


class _Shared(object):
    def __init__(self, value):
        self._value = value

    @property
    def shape(self):
        return self._value.shape

    def __hash__(self):
        return id(self._value)

    def __eq__(self, other):
        return self is other


def _shared(tensor):
    key = '_shared', id(tensor)

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    result = _Shared(tensor)

    cache[key] = result
    return result


def transpose(a, axes):
    backend = _CURRENT_BACKEND[0]
    axes = tuple(axes)
    key = 'transpose', backend, a, axes

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    transpose = get_func('transpose', backend)
    result = _Shared(transpose(a._value, axes))

    cache[key] = result
    return result


def tensordot(x, y, axes=2):
    backend = _CURRENT_BACKEND[0]
    if isinstance(axes, numbers.Number):
        axes = list(range(len(x.shape)))[len(x.shape) - axes:], list(range(len(y.shape)))[:axes]
    axes = tuple(axes[0]), tuple(axes[1])
    key = 'tensordot', backend, x, y, axes

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    tensordot = get_func('tensordot', backend)
    result = _Shared(tensordot(x._value, y._value, axes))

    cache[key] = result
    return result


def einsum(equation, *operands):
    backend = _CURRENT_BACKEND[0]
    # compute a canonical hash, modulo commutativity
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    canonical = sorted(zip(inputs, operands), key=lambda x: id(x[1]))
    canonical_inputs = ','.join(input_ for input_, _ in canonical)
    canonical_equation = _alpha_canonicalize('{}->{}'.format(canonical_inputs, output))
    canonical_operands = tuple(d for _, d in canonical)
    key = 'einsum', backend, canonical_equation, canonical_operands

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    einsum = get_func('einsum', backend)
    result = _Shared(einsum(equation, *(t._value for t in operands)))

    cache[key] = result
    return result
