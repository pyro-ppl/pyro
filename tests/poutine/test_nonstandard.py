from __future__ import absolute_import, division, print_function

from pyro.poutine.nonstandard import NonstandardMessenger, LazyMessenger, LazyValue


def test_nonstandard_simple():

    box = NonstandardMessenger.value_wrapper

    @NonstandardMessenger()
    def model():
        return box(1) + box(2) * box(3)

    x = model()
    assert isinstance(x, box)
    assert x.value == 7


def test_nonstandard_call():

    box = NonstandardMessenger.value_wrapper

    def add(a, b):
        return a + b

    add = box(add, typename="addf")
    NonstandardMessenger.register(fn=lambda msg: print(msg), type="addf")

    @NonstandardMessenger()
    def model():
        return add(box(1), box(2))

    x = model()
    assert isinstance(x, box)
    assert x.value == 3

    NonstandardMessenger.unregister(type="addf")


def test_nonstandard_apply():

    box = NonstandardMessenger.value_wrapper

    def apply_(f, x):
        return f(x)

    apply_ = box(apply_, typename="apply_")
    NonstandardMessenger.register(fn=lambda msg: print(msg), type="apply_")

    @NonstandardMessenger()
    def model():
        return apply_(lambda x: x + 1, box(1))

    x = model()
    assert isinstance(x, box)
    assert x.value == 2


def test_nonstandard_apply_2():

    box = NonstandardMessenger.value_wrapper

    def apply_(f, x):
        return f(x)

    apply_ = box(apply_, typename="apply_")
    NonstandardMessenger.register(fn=lambda msg: print(msg), type="apply_")

    @NonstandardMessenger()
    def model():
        return apply_(box(lambda x: x + 1, typename="add1"), box(1))

    x = model()
    assert isinstance(x, box)
    assert x.value == 2


def test_lazy_simple():

    box = LazyMessenger.value_wrapper

    @LazyMessenger()
    def model():
        return box(1) + box(2) * box(3)

    x = model()
    assert isinstance(x, box)
    assert isinstance(x.value, LazyValue)
    assert x.value.eval().value == 7


def test_mixed_strict_lazy():
    box = LazyMessenger.value_wrapper

    @LazyMessenger()
    def model():
        a = box(1) + box(2) * box(3)
        assert isinstance(a.value, LazyValue)
        b = a.value.eval() + box(1)
        assert isinstance(b.value, LazyValue)
        assert b.value._expr
        c = b + a
        assert isinstance(c.value, LazyValue)
        assert c.value._expr
        return c

    x = model()
    assert isinstance(x.value, LazyValue)
    assert x.value.eval().value == 15
