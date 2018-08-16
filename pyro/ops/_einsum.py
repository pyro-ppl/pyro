from __future__ import absolute_import, division, print_function

import contextlib
import numbers

import opt_einsum

CACHE = {}


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
def shared_intermediates():
    CACHE.clear()
    yield
    CACHE.clear()
