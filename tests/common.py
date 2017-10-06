import contextlib
import os
import sys
import unittest
import numbers
import warnings
from copy import deepcopy
from pytest import approx
from itertools import product
from numpy.testing import assert_allclose

import numpy as np
import torch
import torch.cuda
from torch.autograd import Variable

torch.set_default_tensor_type('torch.DoubleTensor')

"""
Contains test utilities for assertions, approximate comparison (of tensors and other objects).

Code has been largely adapted from pytorch/test/common.py
Source: https://github.com/pytorch/pytorch/blob/master/test/common.py
"""

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(TESTS_DIR, 'resources')
EXAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), 'examples')


def suppress_warnings(fn):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)

    return wrapper


def get_cpu_type(t):
    assert t.__module__ == 'torch.cuda'
    return getattr(torch, t.__class__.__name__)


def get_gpu_type(t):
    assert t.__module__ == 'torch'
    return getattr(torch.cuda, t.__name__)


def to_gpu(obj, type_map={}):
    if torch.is_tensor(obj):
        t = type_map.get(type(obj), get_gpu_type(type(obj)))
        return obj.clone().type(t)
    elif torch.is_storage(obj):
        return obj.new().resize_(obj.size()).copy_(obj)
    elif isinstance(obj, Variable):
        assert obj.is_leaf
        t = type_map.get(type(obj.data), get_gpu_type(type(obj.data)))
        return Variable(obj.data.clone().type(
            t), requires_grad=obj.requires_grad)
    elif isinstance(obj, list):
        return [to_gpu(o, type_map) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(to_gpu(o, type_map) for o in obj)
    else:
        return deepcopy(obj)


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)


def iter_indices(tensor):
    if tensor.dim() == 0:
        return range(0)
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except BaseException:
        return False


def _unwrap_variables(x, y):
    if isinstance(x, Variable) and isinstance(y, Variable):
        return x.data, y.data
    elif isinstance(x, Variable) or isinstance(y, Variable):
        raise AssertionError(
            "cannot compare {} and {}".format(
                type(x), type(y)))
    return x, y


def assert_tensors_equal(a, b, prec=1e-5, msg=''):
    assert a.size() == b.size(), msg
    if a.numel() > 0:
        b = b.type_as(a)
        b = b.cuda(device=a.get_device()) if a.is_cuda else b.cpu()
        # check that NaNs are in the same locations
        nan_mask = a != a
        assert torch.equal(nan_mask, b != b), msg
        diff = a - b
        diff[nan_mask] = 0
        if diff.is_signed():
            diff = diff.abs()
        max_err = diff.max()
        assert max_err < prec, msg


def _safe_coalesce(t):
    tc = t.coalesce()
    value_map = {}
    for idx, val in zip(t._indices().t(), t._values()):
        idx_tup = tuple(idx)
        if idx_tup in value_map:
            value_map[idx_tup] += val
        else:
            value_map[idx_tup] = val.clone() if torch.is_tensor(val) else val

    new_indices = sorted(list(value_map.keys()))
    new_values = [value_map[idx] for idx in new_indices]
    if t._values().ndimension() < 2:
        new_values = t._values().new(new_values)
    else:
        new_values = torch.stack(new_values)

    new_indices = t._indices().new(new_indices).t()
    tg = t.new(new_indices, new_values, t.size())

    assert tc._indices() == tg._indices()
    assert tc._values() == tg._values()
    return tg


def assert_equal(x, y, prec=1e-5, msg=''):
    x, y = _unwrap_variables(x, y)

    if torch.is_tensor(x) and torch.is_tensor(y):
        assert_equal(x.is_sparse, y.is_sparse)
        if x.is_sparse:
            x = _safe_coalesce(x)
            y = _safe_coalesce(y)
            assert_tensors_equal(x._indices(), y._indices(), prec, msg)
            assert_tensors_equal(x._values(), y._values(), prec, msg)
        else:
            assert_tensors_equal(x, y, prec, msg)
    elif type(x) == np.ndarray and type(y) == np.ndarray:
        assert_allclose(x, y, atol=prec, equal_nan=True)
    elif isinstance(x, numbers.Number) and isinstance(y, numbers.Number):
        assert x == approx(y, abs=prec), msg
    elif type(x) != type(y):
        raise AssertionError("cannot compare {} and {}".format(type(x), type(y)))
    elif is_iterable(x) and is_iterable(y):
        assert list(x) == approx(list(y), prec)
    else:
        assert x == y, msg


def assert_not_equal(x, y, prec=1e-5, msg=''):
    try:
        assert_equal(x, y, prec)
    except AssertionError:
        pass
    raise AssertionError("{} \nValues are equal: x={}, y={}, prec={}".format(msg, x, y, prec))


class TestCase(unittest.TestCase):
    precision = 1e-5

    def assertEqual(self, x, y, prec=None, message=''):
        assert_equal(x, y, prec, message)

    def assertNotEqual(self, x, y, prec=None, message=''):
        assert_not_equal(x, y, prec, message)

    def assertObjectIn(self, obj, iterable):
        for elem in iterable:
            if id(obj) == id(elem):
                return
        raise AssertionError("object not found in iterable")

    if sys.version_info < (3, 2):
        # assertRaisesRegexp renamed assertRaisesRegex in 3.2
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


def download_file(url, path, binary=True):
    if sys.version_info < (3,):
        import urllib2
        request = urllib2
        error = urllib2
    else:
        import urllib.request
        import urllib.error
        request = urllib.request
        error = urllib.error

    if os.path.exists(path):
        return True
    try:
        data = request.urlopen(url, timeout=15).read()
        with open(path, 'wb' if binary else 'w') as f:
            f.write(data)
        return True
    except error.URLError:
        return False
