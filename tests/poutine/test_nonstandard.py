from __future__ import absolute_import, division, print_function

from pyro.poutine.nonstandard import Box, NonstandardMessenger, LazyMessenger, LazyValue


def test_nonstandard_simple():

    @NonstandardMessenger()
    def model():
        return Box(1) + Box(2) * Box(3)

    x = model()
    assert isinstance(x, Box)
    assert x.value == 7


def test_nonstandard_call():

    def add(a, b):
        return a + b

    add = Box(add, typename="addf")
    NonstandardMessenger.register(fn=lambda msg: print(msg), type="addf")

    @NonstandardMessenger()
    def model():
        return add(Box(1), Box(2))

    x = model()
    assert isinstance(x, Box)
    assert x.value == 3

    NonstandardMessenger.unregister(type="addf")


def test_nonstandard_apply():

    def apply_(f, x):
        return f(x)

    apply_ = Box(apply_, typename="apply_")
    NonstandardMessenger.register(fn=lambda msg: print(msg), type="apply_")

    @NonstandardMessenger()
    def model():
        return apply_(lambda x: x + 1, Box(1))


def test_nonstandard_apply_2():

    def apply_(f, x):
        return f(x)

    apply_ = Box(apply_, typename="apply_")
    NonstandardMessenger.register(fn=lambda msg: print(msg), type="apply_")

    @NonstandardMessenger()
    def model():
        return apply_(Box(lambda x: x + 1, typename="add1"), Box(1))

    x = model()
    assert isinstance(x, Box)
    assert x.value == 2


def test_lazy_simple():

    @LazyMessenger()
    def model():
        return Box(1) + Box(2) * Box(3)

    x = model()
    assert isinstance(x, Box)
    assert isinstance(x.value, LazyValue)
    assert x.eval().value == 7
    assert x.value.eval() == 7


def test_mixed_strict_lazy():

    @LazyMessenger()
    def model():
        a = Box(1) + Box(2) * Box(3)
        assert isinstance(a.value, LazyValue)
        b = a.eval() + Box(1)
        assert isinstance(b.value, LazyValue)
        assert b.value._expr
        c = b + a
        assert isinstance(c.value, LazyValue)
        assert c.value._expr
        return c

    x = model()
    assert isinstance(x.value, LazyValue)
    assert x.eval().value == 15
    assert x.value.eval() == 15
