from __future__ import absolute_import, division, print_function

import contextlib
import numbers
from collections import OrderedDict

import opt_einsum

from pyro.distributions.torch_patch import _patch

CACHE = OrderedDict()


class Deferred(object):
    def __init__(self, shape):
        self.shape = shape
        self._value = None

    def __eq__(self, other):
        return self is other

    def eval(self):
        if self._value is None:
            self._eval()
            assert self._value.shape == self.shape
        return self._value


class DeferredTensor(Deferred):
    def __init__(self, tensor):
        super(DeferredTensor, self).__init__(tensor.shape)
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
        self.a = a
        self.axes = tuple(axes)
        shape = tuple(a.shape[i] for i in axes)
        super(Transpose, self).__init__(shape)

    def _eval(self):
        a = self.a.eval()
        self._value = opt_einsum.backends.torch.transpose(a, self.axes)

    def __hash__(self):
        return hash((self.a, self.axes))


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
        self.x = x
        self.y = y
        self.axes = axes
        x_shape = tuple(s for i, s in enumerate(x.shape) if i not in axes[0])
        y_shape = tuple(s for i, s in enumerate(y.shape) if i not in axes[1])
        shape = x_shape + y_shape
        super(Tensordot, self).__init__(shape)

    def _eval(self):
        x = self.x.eval()
        y = self.y.eval()

        # This workaround can be deleted after this issue is fixed in release:
        # https://github.com/pytorch/pytorch/issues/7763
        x, y = x.clone(), y.clone()

        self._value = opt_einsum.backends.torch.tensordot(x, y, self.axes)

    def __hash__(self):
        return hash((self.x, self.y, self.axes))


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
        self.equation = equation
        self.operands = operands
        inputs, output = equation.split('->')
        inputs = inputs.split(',')
        assert len(inputs) == len(operands)
        sizes = {}
        for names, tensor in zip(inputs, operands):
            assert len(names) == len(tensor.shape)
            for name, size in zip(names, tensor.shape):
                sizes[name] = size
        shape = tuple(sizes[name] for name in output)
        super(Einsum, self).__init__(shape)

    def _eval(self):
        operands = [d.eval() for d in self.operands]

        # This workaround can be deleted after this issue is fixed in release:
        # https://github.com/pytorch/pytorch/issues/7763
        operands = [d.clone() for d in operands]

        self._value = opt_einsum.backends.torch.einsum(self.equation, *operands)

    def __hash__(self):
        return hash((self.equation, self.operands))


def einsum(equation, *operands):
    operands = tuple(operands)

    key = 'einsum', equation, operands
    if key in CACHE:
        return CACHE[key]

    result = Einsum(equation, operands)
    CACHE[key] = result
    return result


@contextlib.contextmanager
def shared_intermediates(debug=False):
    CACHE.clear()
    yield

    if debug:
        names = {value: 'x{}'.format(i) for i, value in enumerate(CACHE.values())}
        for value in CACHE.values():
            if isinstance(value, DeferredTensor):
                args = '...'
            elif isinstance(value, Transpose):
                args = '{}, {}'.format(names[value.a], value.axes)
            elif isinstance(value, Tensordot):
                args = '{}, {}, {}'.format(names[value.x], names[value.y], value.axes)
            elif isinstance(value, Einsum):
                args = '{}, {}'.format(value.equation, ', '.join(names[o] for o in value.operands))
            print('{} = {}({})'.format(names[value], type(value).__name__, args))

    CACHE.clear()


# Work around torch.einsum's limitation to 26 letters
@_patch('torch.einsum')
def _einsum(equation, operands):

    # attempt to alpha convert to a-z
    target = 'abcdefghijklmnopqrstuvwxyz'
    source = sorted(set(name for name in equation if name not in ',->'))
    rename = dict(zip(source, target))
    equation = ''.join(rename.get(x, x) for x in equation)

    return _einsum._pyro_unpatched(equation, operands)
