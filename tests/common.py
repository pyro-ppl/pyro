from __future__ import absolute_import, division, print_function

import contextlib
import numbers
import os
import warnings
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch
import torch.cuda
from numpy.testing import assert_allclose
from pytest import approx
from torch.autograd import Variable

torch.set_default_tensor_type(os.environ.get('PYRO_TENSOR_TYPE', 'torch.DoubleTensor'))

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


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="cuda is not available")


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
def tensors_default_to(host):
    """
    Context manager to temporarily use Cpu or Cuda tensors in Pytorch.

    :param str host: Either "cuda" or "cpu".
    """
    assert host in ('cpu', 'cuda'), host
    old_module = torch.Tensor.__module__
    name = torch.Tensor.__name__
    new_module = 'torch.cuda' if host == 'cuda' else 'torch'
    torch.set_default_tensor_type('{}.{}'.format(new_module, name))
    try:
        yield
    finally:
        torch.set_default_tensor_type('{}.{}'.format(old_module, name))


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)


@contextlib.contextmanager
def xfail_if_not_implemented(msg="Not implemented"):
    try:
        yield
    except NotImplementedError as e:
        pytest.xfail(reason="{}: {}".format(msg, e))


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
    if prec == 0:
        assert (a == b).all(), msg
    elif a.numel() > 0:
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


# TODO Split this into assert_equal() and assert_close() or assert_almost_equal().
# TODO Use atol and rtol instead of prec
def assert_equal(x, y, prec=1e-5, msg=''):
    x, y = _unwrap_variables(x, y)

    if torch.is_tensor(x) and torch.is_tensor(y):
        assert_equal(x.is_sparse, y.is_sparse, prec, msg)
        if x.is_sparse:
            x = _safe_coalesce(x)
            y = _safe_coalesce(y)
            assert_tensors_equal(x._indices(), y._indices(), prec, msg)
            assert_tensors_equal(x._values(), y._values(), prec, msg)
        else:
            assert_tensors_equal(x, y, prec, msg)
    elif type(x) == np.ndarray and type(y) == np.ndarray:
        if prec == 0:
            assert (x == y).all(), msg
        else:
            assert_allclose(x, y, atol=prec, equal_nan=True)
    elif isinstance(x, numbers.Number) and isinstance(y, numbers.Number):
        if not msg:
            msg = '{} vs {}'.format(x, y)
        if prec == 0:
            assert x == y, msg
        else:
            assert x == approx(y, abs=prec), msg
    elif type(x) != type(y):
        raise AssertionError("cannot compare {} and {}".format(type(x), type(y)))
    elif isinstance(x, str):
        assert x == y, msg
    elif isinstance(x, dict):
        assert set(x.keys()) == set(y.keys())
        for key, x_val in x.items():
            assert_equal(x_val, y[key], prec, msg='{} {}'.format(key, msg))
    elif is_iterable(x) and is_iterable(y):
        if prec == 0:
            assert len(x) == len(y), msg
            for xi, yi in zip(x, y):
                assert_equal(xi, yi, prec, msg)
        else:
            if not msg:
                msg = '{} vs {}'.format(x, y)
            assert list(x) == approx(list(y), prec), msg
    else:
        assert x == y, msg


def assert_not_equal(x, y, prec=1e-5, msg=''):
    try:
        assert_equal(x, y, prec)
    except AssertionError:
        pass
    raise AssertionError("{} \nValues are equal: x={}, y={}, prec={}".format(msg, x, y, prec))
