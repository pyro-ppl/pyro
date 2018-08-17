from __future__ import absolute_import, division, print_function

import contextlib
import numbers
from collections import OrderedDict

import opt_einsum

from pyro.distributions.torch_patch import _patch

CACHE = OrderedDict()
NEXT_SYMBOL = [0]
LAST_CACHE_SIZE = [0]
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


@contextlib.contextmanager
def shared_intermediates(debug=False):
    CACHE.clear()
    NEXT_SYMBOL[0] = 0
    yield CACHE

    if debug:
        for i, value in enumerate(CACHE.values()):
            if isinstance(value, DeferredTensor):
                rhs = 'tensor{}'.format(tuple(value._value.shape))
            elif isinstance(value, Transpose):
                rhs = str(value)
            elif isinstance(value, Tensordot):
                rhs = '{} * {}'.format(value._x, value._y)
            elif isinstance(value, Einsum):
                rhs = ' * '.join(str(d) for d in value._operands)
            dims = '?' if value._dims is None else value._dims
            print('{: >4} {: <14}{: >8} {} = {}'.format(
                i, type(value).__name__, dims, value, rhs))

    LAST_CACHE_SIZE[0] = len(CACHE)
    CACHE.clear()
    NEXT_SYMBOL[0] = 0


def contract(equation, *operands, **kwargs):
    """
    Like :func:`opt_einsum.contract` but adds debugging metadata.
    """
    # add debugging metadata
    if kwargs.get('backend', 'torch') == 'pyro.ops._einsum':
        inputs, output = equation.split('->')
        inputs = inputs.split(',')
        for dims, d in zip(inputs, operands):
            d._dims = dims

    return opt_einsum.contract(equation, *operands, **kwargs)


def alpha_canonicalize(equation):
    """
    Attempt to alpha convert to an equation to the letters a-z,
    in an order-independent canonical way.
    """
    rename = OrderedDict()
    for name in equation:
        if name in ',->':
            continue
        if name not in rename:
            rename[name] = ALPHABET[len(rename)]
    return ''.join(rename.get(x, x) for x in equation)


class Deferred(object):
    def __init__(self, shape, name):
        self.shape = shape
        self._value = None
        self._name = name  # debugging info
        self._dims = None  # debugging info set by contract()

    def __str__(self):
        return self._name

    def __eq__(self, other):
        return self is other

    def eval(self):
        if self._value is None:
            self._eval()
            assert self._value.shape == self.shape
        return self._value


class DeferredTensor(Deferred):
    def __init__(self, tensor):
        name = opt_einsum.get_symbol(NEXT_SYMBOL[0])
        NEXT_SYMBOL[0] += 1
        super(DeferredTensor, self).__init__(tensor.shape, name)
        self._value = tensor

    def __hash__(self):
        return id(self._value)

    def squeeze(self):
        key = 'squeeze', self
        if key in CACHE:
            return CACHE[key]

        result = DeferredTensor(self._value.squeeze())
        CACHE[key] = result
        return result


def deferred_tensor(tensor):
    key = 'deferred_tensor', id(tensor)
    if key in CACHE:
        return CACHE[key]

    result = DeferredTensor(tensor)
    CACHE[key] = result
    return result


class Transpose(Deferred):
    def __init__(self, a, axes):
        assert isinstance(a, Deferred)
        self._a = a
        self._axes = tuple(axes)
        shape = tuple(a.shape[i] for i in axes)
        name = a._name + "'"
        super(Transpose, self).__init__(shape, name)
        if a._dims is not None:
            self._dims = ''.join(a._dims[i] for i in axes)

    def _eval(self):
        a = self._a.eval()
        self._value = opt_einsum.backends.torch.transpose(a, self._axes)

    def __hash__(self):
        return hash((self._a, self._axes))


def transpose(a, axes):
    axes = tuple(axes)

    key = 'transpose', a, axes
    if key in CACHE:
        return CACHE[key]

    result = Transpose(a, axes)
    CACHE[key] = result
    return result


class Tensordot(Deferred):
    def __init__(self, x, y, axes):
        assert isinstance(x, Deferred)
        assert isinstance(y, Deferred)
        self._x = x
        self._y = y
        self._axes = axes
        x_shape = tuple(s for i, s in enumerate(x.shape) if i not in axes[0])
        y_shape = tuple(s for i, s in enumerate(y.shape) if i not in axes[1])
        shape = x_shape + y_shape
        name = '({}{})'.format(x._name, y._name)
        super(Tensordot, self).__init__(shape, name)
        if x._dims is not None and y._dims is not None:
            self._dims = ''.join([s for i, s in enumerate(x._dims) if i not in axes[0]] +
                                 [s for i, s in enumerate(y._dims) if i not in axes[1]])

    def _eval(self):
        x = self._x.eval()
        y = self._y.eval()

        # This workaround can be deleted after this issue is fixed in release:
        # https://github.com/pytorch/pytorch/issues/7763
        x, y = x.clone(), y.clone()

        self._value = opt_einsum.backends.torch.tensordot(x, y, self._axes)

    def __hash__(self):
        return hash((self._x, self._y, self._axes))


def tensordot(x, y, axes=2):
    if isinstance(axes, numbers.Number):
        axes = list(range(len(x.shape)))[len(x.shape) - axes:], list(range(len(y.shape)))[:axes]
    axes = tuple(axes[0]), tuple(axes[1])

    key = 'tensordot', x, y, axes
    if key in CACHE:
        return CACHE[key]

    result = Tensordot(x, y, axes)
    CACHE[key] = result
    return result


class Einsum(Deferred):
    def __init__(self, equation, operands):
        assert all(isinstance(d, Deferred) for d in operands)
        self._equation = equation
        self._operands = tuple(operands)
        inputs, output = equation.split('->')
        inputs = inputs.split(',')
        assert len(inputs) == len(operands)
        sizes = {}
        for names, tensor in zip(inputs, operands):
            assert len(names) == len(tensor.shape)
            for name, size in zip(names, tensor.shape):
                sizes[name] = size
        shape = tuple(sizes[name] for name in output)
        name = '({})'.format(''.join(str(d) for d in operands))
        super(Einsum, self).__init__(shape, name)
        if all(d._dims is not None for d in operands):
            self._dims = output

    def _eval(self):
        operands = [d.eval() for d in self._operands]

        # This workaround can be deleted after this issue is fixed in release:
        # https://github.com/pytorch/pytorch/issues/7763
        operands = [d.clone() for d in operands]

        self._value = opt_einsum.backends.torch.einsum(self._equation, *operands)

    def __hash__(self):
        return hash((self._equation, self._operands))


def einsum(equation, *operands):
    # compute a canonical hash, modulo commutativity
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    canonical = sorted(zip(inputs, operands), key=lambda x: id(x[1]))
    canonical_inputs = ','.join(input_ for input_, _ in canonical)
    canonical_equation = alpha_canonicalize('{}->{}'.format(canonical_inputs, output))
    canonical_operands = tuple(d for _, d in canonical)

    key = 'einsum', canonical_equation, canonical_operands
    if key in CACHE:
        return CACHE[key]

    result = Einsum(equation, operands)
    CACHE[key] = result
    return result


# Work around torch.einsum's limitation to 26 letters
@_patch('torch.einsum')
def _einsum(equation, operands):
    equation = alpha_canonicalize(equation)
    return _einsum._pyro_unpatched(equation, operands)
