# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import numbers
import os
import shutil
import tempfile
import warnings
from itertools import product

import numpy as np
import pytest
import torch
import torch.cuda
from numpy.testing import assert_allclose
from pytest import approx

"""
Contains test utilities for assertions, approximate comparison (of tensors and other objects).

Code has been largely adapted from pytorch/test/common.py
Source: https://github.com/pytorch/pytorch/blob/master/test/common.py
"""

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(TESTS_DIR, 'resources')
EXAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), 'examples')


def xfail_param(*args, **kwargs):
    return pytest.param(*args, marks=[pytest.mark.xfail(**kwargs)])


def skipif_param(*args, **kwargs):
    return pytest.param(*args, marks=[pytest.mark.skipif(**kwargs)])


def suppress_warnings(fn):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)

    return wrapper


# backport of Python 3's context manager
@contextlib.contextmanager
def TemporaryDirectory():
    try:
        path = tempfile.mkdtemp()
        yield path
    finally:
        if os.path.exists(path):
            shutil.rmtree(path)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="cuda is not available")


def get_cpu_type(t):
    assert t.__module__ == 'torch.cuda'
    return getattr(torch, t.__class__.__name__)


def get_gpu_type(t):
    assert t.__module__ == 'torch'
    return getattr(torch.cuda, t.__name__)


@contextlib.contextmanager
def tensors_default_to(host):
    """
    Context manager to temporarily use Cpu or Cuda tensors in PyTorch.

    :param str host: Either "cuda" or "cpu".
    """
    assert host in ('cpu', 'cuda'), host
    old_module, name = torch.Tensor().type().rsplit('.', 1)
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


def assert_tensors_equal(a, b, prec=0., msg=''):
    assert a.size() == b.size(), msg
    if isinstance(prec, numbers.Number) and prec == 0:
        assert (a == b).all(), msg
    if a.numel() == 0 and b.numel() == 0:
        return
    b = b.type_as(a)
    b = b.cuda(device=a.get_device()) if a.is_cuda else b.cpu()
    # check that NaNs are in the same locations
    nan_mask = a != a
    assert torch.equal(nan_mask, b != b), msg
    diff = a - b
    diff[a == b] = 0  # handle inf
    diff[nan_mask] = 0
    if diff.is_signed():
        diff = diff.abs()
    if isinstance(prec, torch.Tensor):
        assert (diff <= prec).all(), msg
    else:
        max_err = diff.max().item()
        assert (max_err <= prec), msg


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
    if t._values().dim() < 2:
        new_values = t._values().new_tensor(new_values)
    else:
        new_values = torch.stack(new_values)

    new_indices = t._indices().new_tensor(new_indices).t()
    tg = t.new(new_indices, new_values, t.size())

    assert (tc._indices() == tg._indices()).all()
    assert (tc._values() == tg._values()).all()
    return tg


def assert_close(actual, expected, atol=1e-7, rtol=0, msg=''):
    if not msg:
        msg = '{} vs {}'.format(actual, expected)
    if isinstance(actual, numbers.Number) and isinstance(expected, numbers.Number):
        assert actual == approx(expected, abs=atol, rel=rtol), msg
    # Placing this as a second check allows for coercing of numeric types above;
    # this can be moved up to harden type checks.
    elif type(actual) != type(expected):
        raise AssertionError("cannot compare {} and {}".format(type(actual),
                                                               type(expected)))
    elif torch.is_tensor(actual) and torch.is_tensor(expected):
        prec = atol + rtol * abs(expected) if rtol > 0 else atol
        assert actual.is_sparse == expected.is_sparse, msg
        if actual.is_sparse:
            x = _safe_coalesce(actual)
            y = _safe_coalesce(expected)
            assert_tensors_equal(x._indices(), y._indices(), prec, msg)
            assert_tensors_equal(x._values(), y._values(), prec, msg)
        else:
            assert_tensors_equal(actual, expected, prec, msg)
    elif type(actual) == np.ndarray and type(expected) == np.ndarray:
        assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True, err_msg=msg)
    elif isinstance(actual, numbers.Number) and isinstance(y, numbers.Number):
        assert actual == approx(expected, abs=atol, rel=rtol), msg
    elif isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key, x_val in actual.items():
            assert_close(x_val, expected[key], atol=atol, rtol=rtol,
                         msg='At key{}: {} vs {}'.format(key, x_val, expected[key]))
    elif isinstance(actual, str):
        assert actual == expected, msg
    elif is_iterable(actual) and is_iterable(expected):
        assert len(actual) == len(expected), msg
        for xi, yi in zip(actual, expected):
            assert_close(xi, yi, atol=atol, rtol=rtol, msg='{} vs {}'.format(xi, yi))
    else:
        assert actual == expected, msg


# TODO: Remove `prec` arg, and move usages to assert_close
def assert_equal(actual, expected, prec=1e-5, msg=''):
    if prec > 0.:
        return assert_close(actual, expected, atol=prec, msg=msg)
    if not msg:
        msg = '{} vs {}'.format(actual, expected)
    if isinstance(actual, numbers.Number) and isinstance(expected, numbers.Number):
        assert actual == expected, msg
    # Placing this as a second check allows for coercing of numeric types above;
    # this can be moved up to harden type checks.
    elif type(actual) != type(expected):
        raise AssertionError("cannot compare {} and {}".format(type(actual),
                                                               type(expected)))
    elif torch.is_tensor(actual) and torch.is_tensor(expected):
        assert actual.is_sparse == expected.is_sparse, msg
        if actual.is_sparse:
            x = _safe_coalesce(actual)
            y = _safe_coalesce(expected)
            assert_tensors_equal(x._indices(), y._indices(), msg=msg)
            assert_tensors_equal(x._values(), y._values(), msg=msg)
        else:
            assert_tensors_equal(actual, expected, msg=msg)
    elif type(actual) == np.ndarray and type(actual) == np.ndarray:
        assert (actual == expected).all(), msg
    elif isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key, x_val in actual.items():
            assert_equal(x_val, expected[key], prec=0.,
                         msg='At key{}: {} vs {}'.format(key, x_val, expected[key]))
    elif isinstance(actual, str):
        assert actual == expected, msg
    elif is_iterable(actual) and is_iterable(expected):
        assert len(actual) == len(expected), msg
        for xi, yi in zip(actual, expected):
            assert_equal(xi, yi, prec=0., msg='{} vs {}'.format(xi, yi))
    else:
        assert actual == expected, msg


def assert_not_equal(x, y, prec=1e-5, msg=''):
    try:
        assert_equal(x, y, prec)
    except AssertionError:
        return
    raise AssertionError("{} \nValues are equal: x={}, y={}, prec={}".format(msg, x, y, prec))
