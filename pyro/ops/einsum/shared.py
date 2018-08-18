from __future__ import absolute_import, division, print_function

import contextlib
import numbers
from collections import OrderedDict

import opt_einsum
from pyro.distributions.torch_patch import _patch

_SHARING_STACK = []
_CURRENT_BACKEND = []
LAST_CACHE_SIZE = [0]  # for profiling


@contextlib.contextmanager
def shared_intermediates(cache=None):
    """
    Context in which :func:`contract` intermediate results are shared.
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
    Like :func:`opt_einsum.contract` but works with shared_intermediates contexts.
    """
    backend = kwargs.pop('backend', 'numpy')

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
        raise ValueError('sharing backend is available only via shared_intermediates')

    return opt_einsum.contract(equation, *operands, backend=backend, **kwargs)


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

    def squeeze(self):
        key = 'squeeze', self

        cache = _SHARING_STACK[-1]
        if key in cache:
            return cache[key]

        result = _Shared(self._value.squeeze())

        cache[key] = result
        return result


def _shared(tensor):
    key = '_shared', id(tensor)

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    result = _Shared(tensor)

    cache[key] = result
    return result


def transpose(a, axes):
    axes = tuple(axes)
    key = 'transpose', a, axes

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    a = a._value
    transpose = opt_einsum.backends.dispatch.get_func('transpose', _CURRENT_BACKEND[0])
    result = _Shared(transpose(a, axes))

    cache[key] = result
    return result


def tensordot(x, y, axes=2):
    if isinstance(axes, numbers.Number):
        axes = list(range(len(x.shape)))[len(x.shape) - axes:], list(range(len(y.shape)))[:axes]
    axes = tuple(axes[0]), tuple(axes[1])
    key = 'tensordot', x, y, axes

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    x = x._value
    y = y._value

    # This workaround can be deleted after this issue is fixed in release:
    # https://github.com/pytorch/pytorch/issues/7763
    x, y = x.clone(), y.clone()

    tensordot = opt_einsum.backends.dispatch.get_func('tensordot', _CURRENT_BACKEND[0])
    result = _Shared(tensordot(x, y, axes))

    cache[key] = result
    return result


def einsum(equation, *operands):
    # compute a canonical hash, modulo commutativity
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    canonical = sorted(zip(inputs, operands), key=lambda x: id(x[1]))
    canonical_inputs = ','.join(input_ for input_, _ in canonical)
    canonical_equation = _alpha_canonicalize('{}->{}'.format(canonical_inputs, output))
    canonical_operands = tuple(d for _, d in canonical)
    key = 'einsum', canonical_equation, canonical_operands

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    operands = [t._value for t in operands]

    # This workaround can be deleted after this issue is fixed in release:
    # https://github.com/pytorch/pytorch/issues/7763
    operands = [t.clone() for t in operands]

    einsum = opt_einsum.backends.dispatch.get_func('einsum', _CURRENT_BACKEND[0])
    result = _Shared(einsum(equation, *operands))

    cache[key] = result
    return result


# Work around torch.einsum's limitation to 26 letters
@_patch('torch.einsum')
def _einsum(equation, operands):
    equation = _alpha_canonicalize(equation)
    return _einsum._pyro_unpatched(equation, operands)
