from __future__ import absolute_import, division, print_function

import functools
from unittest import TestCase

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.poutine.nonstandard import NonstandardMessenger, LazyMessenger, LazyValue


def test_nonstandard_simple():
    
    box = NonstandardMessenger.value_wrapper

    @NonstandardMessenger()
    def model():
        return box(1) + box(2) * box(3)

    model()


def test_lazy_simple():

    box = LazyMessenger.value_wrapper

    @LazyMessenger()
    def model():
        return box(1) + box(2) * box(3)

    x = model()
    assert isinstance(x.value, LazyValue)
    assert x.value.eval() == 7


def test_mixed_strict_lazy():
    box = LazyMessenger.value_wrapper

    @LazyMessenger()
    def model():
        a = box(1) + box(2) * box(3)
        assert isinstance(a.value, LazyValue)
        b = box(a.value.eval()) + box(1)
        assert isinstance(b.value, LazyValue)
        assert b.value._expr
        c = b + a
        assert isinstance(c.value, LazyValue)
        assert c.value._expr
        return c

    x = model()
    assert isinstance(x.value, LazyValue)
    assert x.value.eval() == 15
