# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyro.infer.autoguide.utils import deep_getattr, deep_setattr
from pyro.nn import PyroModule


def test_deep_getattr_setattr_roundtrip():
    module = PyroModule()
    deep_setattr(module, "a.b.c", 1.5)
    assert deep_getattr(module, "a.b.c") == 1.5
    assert deep_getattr(module, "a.b") is module.a.b


def test_deep_getattr_missing():
    with pytest.raises(AttributeError):
        deep_getattr(PyroModule(), "a.b")
