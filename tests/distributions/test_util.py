# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import weakref

import numpy as np
import pytest
import torch
from torch.autograd import grad

import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.distributions.util import (
    broadcast_shape,
    deep_to,
    detach,
    sum_leftmost,
    sum_rightmost,
    weakmethod,
)
from pyro.nn.module import PyroModule, PyroParam, PyroSample
from tests.common import assert_equal

INF = float("inf")


@pytest.mark.parametrize(
    "shapes",
    [
        ([],),
        ([1],),
        ([2],),
        ([], []),
        ([], [1]),
        ([], [2]),
        ([1], []),
        ([2], []),
        ([1], [2]),
        ([2], [1]),
        ([2], [2]),
        ([2], [3, 1]),
        ([2, 1], [3]),
        ([2, 1], [1, 3]),
        ([1, 2, 4, 1, 3], [6, 7, 1, 1, 5, 1]),
        ([], [3, 1], [2], [4, 3, 1], [5, 4, 1, 1]),
    ],
)
def test_broadcast_shape(shapes):
    assert broadcast_shape(*shapes) == np.broadcast(*map(np.empty, shapes)).shape


@pytest.mark.parametrize(
    "shapes",
    [
        ([3], [4]),
        ([2, 1], [1, 3, 1]),
    ],
)
def test_broadcast_shape_error(shapes):
    with pytest.raises((ValueError, RuntimeError)):
        broadcast_shape(*shapes)


@pytest.mark.parametrize(
    "shapes",
    [
        ([],),
        ([1],),
        ([2],),
        ([], []),
        ([], [1]),
        ([], [2]),
        ([1], []),
        ([2], []),
        ([1], [1]),
        ([2], [2]),
        ([2], [2]),
        ([2], [3, 2]),
        ([2, 3], [3]),
        ([2, 3], [2, 3]),
        ([4], [1, 2, 3, 4], [2, 3, 4], [3, 4]),
    ],
)
def test_broadcast_shape_strict(shapes):
    assert (
        broadcast_shape(*shapes, strict=True)
        == np.broadcast(*map(np.empty, shapes)).shape
    )


@pytest.mark.parametrize(
    "shapes",
    [
        ([1], [2]),
        ([2], [1]),
        ([3], [4]),
        ([2], [3, 1]),
        ([2, 1], [3]),
        ([2, 1], [1, 3]),
        ([2, 1], [1, 3, 1]),
        ([1, 2, 4, 1, 3], [6, 7, 1, 1, 5, 1]),
        ([], [3, 1], [2], [4, 3, 1], [5, 4, 1, 1]),
    ],
)
def test_broadcast_shape_strict_error(shapes):
    with pytest.raises(ValueError):
        broadcast_shape(*shapes, strict=True)


def test_sum_rightmost():
    x = torch.ones(2, 3, 4)
    assert sum_rightmost(x, 0).shape == (2, 3, 4)
    assert sum_rightmost(x, 1).shape == (2, 3)
    assert sum_rightmost(x, 2).shape == (2,)
    assert sum_rightmost(x, -1).shape == (2,)
    assert sum_rightmost(x, -2).shape == (2, 3)
    assert sum_rightmost(x, INF).shape == ()


def test_sum_leftmost():
    x = torch.ones(2, 3, 4)
    assert sum_leftmost(x, 0).shape == (2, 3, 4)
    assert sum_leftmost(x, 1).shape == (3, 4)
    assert sum_leftmost(x, 2).shape == (4,)
    assert sum_leftmost(x, -1).shape == (4,)
    assert sum_leftmost(x, -2).shape == (3, 4)
    assert sum_leftmost(x, INF).shape == ()


def test_weakmethod():
    class Foo:
        def __init__(self, state):
            self.state = state
            self.method = self._method

        @weakmethod
        def _method(self, *args, **kwargs):
            return self.state, args, kwargs

    foo = Foo(42)
    assert foo.method(1, 2, 3, x=0) == (42, (1, 2, 3), {"x": 0})

    foo_ref = weakref.ref(foo)
    assert foo_ref() is foo
    del foo
    assert foo_ref() is None


@pytest.mark.parametrize("shape", [None, (), (4,), (3, 2)], ids=str)
def test_detach_normal(shape):
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    d1 = dist.Normal(loc, scale)
    if shape is not None:
        d1 = d1.expand(shape)

    d2 = detach(d1)
    assert type(d1) is type(d2)
    assert_equal(d1.loc, d2.loc)
    assert_equal(d1.scale, d2.scale)
    assert not d2.loc.requires_grad
    assert not d2.scale.requires_grad


@pytest.mark.parametrize("shape", [None, (), (4,), (3, 2)], ids=str)
def test_detach_beta(shape):
    concentration1 = torch.tensor(0.5, requires_grad=True)
    concentration0 = torch.tensor(2.0, requires_grad=True)
    d1 = dist.Beta(concentration1, concentration0)
    if shape is not None:
        d1 = d1.expand(shape)

    d2 = detach(d1)
    assert type(d1) is type(d2)
    assert d2.batch_shape == d1.batch_shape
    assert_equal(d1.concentration1, d2.concentration1)
    assert_equal(d1.concentration0, d2.concentration0)
    assert not d2.concentration1.requires_grad
    assert not d2.concentration0.requires_grad


@pytest.mark.parametrize("shape", [None, (), (4,), (3, 2)], ids=str)
def test_detach_transformed(shape):
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    d1 = dist.TransformedDistribution(
        dist.Normal(loc, scale), dist.transforms.AffineTransform(a, b)
    )
    if shape is not None:
        d1 = d1.expand(shape)

    d2 = detach(d1)
    assert type(d1) is type(d2)
    assert d2.event_shape == d1.event_shape
    assert d2.batch_shape == d1.batch_shape
    assert type(d1.base_dist) is type(d2.base_dist)
    assert len(d1.transforms) == len(d2.transforms)
    assert_equal(d1.base_dist.loc, d2.base_dist.loc)
    assert_equal(d1.base_dist.scale, d2.base_dist.scale)
    assert_equal(d1.transforms[0].loc, d2.transforms[0].loc)
    assert_equal(d1.transforms[0].scale, d2.transforms[0].scale)
    assert not d2.base_dist.loc.requires_grad
    assert not d2.base_dist.scale.requires_grad
    assert not d2.transforms[0].loc.requires_grad
    assert not d2.transforms[0].scale.requires_grad


@pytest.mark.parametrize("shape", [None, (), (4,), (3, 2)], ids=str)
def test_detach_jit(shape):
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    data = torch.randn(5, 1, 1)

    def fn(loc, scale, data):
        d = dist.Normal(loc, scale, validate_args=False)
        if shape is not None:
            d = d.expand(shape)
        return detach(d).log_prob(data)

    jit_fn = torch.jit.trace(fn, (loc, scale, data))

    expected = fn(loc, scale, data)
    actual = jit_fn(loc, scale, data)
    assert not expected.requires_grad
    assert not actual.requires_grad
    assert_equal(actual, expected)


@pytest.mark.parametrize("shape", [None, (), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
def test_deep_to_normal(shape, dtype):
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=False)
    d1 = dist.Normal(loc, scale)
    if shape is not None:
        d1 = d1.expand(shape)

    # Pass dtype positionally.
    d2 = deep_to(d1, dtype)
    d2.log_prob(d2.sample().detach())  # smoke test
    assert type(d1) is type(d2)
    assert d2.loc.dtype == dtype
    assert d2.scale.dtype == dtype
    assert_equal(d1.loc.to(dtype), d2.loc)
    assert_equal(d1.scale.to(dtype), d2.scale)
    assert d2.loc.requires_grad
    assert not d2.scale.requires_grad

    # Pass dtype by keyword.
    d2 = deep_to(d1, dtype=dtype)
    d2.log_prob(d2.sample().detach())  # smoke test
    assert type(d1) is type(d2)
    assert d2.loc.dtype == dtype
    assert d2.scale.dtype == dtype
    assert_equal(d1.loc.to(dtype), d2.loc)
    assert_equal(d1.scale.to(dtype), d2.scale)
    assert d2.loc.requires_grad
    assert not d2.scale.requires_grad


@pytest.mark.parametrize("shape", [None, (), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
def test_deep_to_beta(shape, dtype):
    concentration1 = torch.tensor(0.5, requires_grad=True)
    concentration0 = torch.tensor(2.0, requires_grad=False)
    d1 = dist.Beta(concentration1, concentration0)
    if shape is not None:
        d1 = d1.expand(shape)

    d2 = deep_to(d1, dtype)
    d2.log_prob(d2.sample().detach())  # smoke test
    assert type(d1) is type(d2)
    assert d2.batch_shape == d1.batch_shape
    assert_equal(d1.concentration1.to(dtype), d2.concentration1)
    assert_equal(d1.concentration0.to(dtype), d2.concentration0)
    assert d2.concentration1.requires_grad == d1.concentration1.requires_grad
    assert d2.concentration0.requires_grad == d1.concentration0.requires_grad


@pytest.mark.parametrize("shape", [None, (), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
def test_deep_to_transformed(shape, dtype):
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=False)
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=False)
    d1 = dist.TransformedDistribution(
        dist.Normal(loc, scale), dist.transforms.AffineTransform(a, b)
    )
    if shape is not None:
        d1 = d1.expand(shape)

    d2 = deep_to(d1, dtype)
    d2.log_prob(d2.sample().detach())  # smoke test
    assert type(d1) is type(d2)
    assert d2.event_shape == d1.event_shape
    assert d2.batch_shape == d1.batch_shape
    assert type(d1.base_dist) is type(d2.base_dist)
    assert len(d1.transforms) == len(d2.transforms)
    assert_equal(d1.base_dist.loc.to(dtype), d2.base_dist.loc)
    assert_equal(d1.base_dist.scale.to(dtype), d2.base_dist.scale)
    assert_equal(d1.transforms[0].loc.to(dtype), d2.transforms[0].loc)
    assert_equal(d1.transforms[0].scale.to(dtype), d2.transforms[0].scale)
    assert d2.base_dist.loc.requires_grad
    assert not d2.base_dist.scale.requires_grad
    assert d2.transforms[0].loc.requires_grad
    assert not d2.transforms[0].scale.requires_grad


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
def test_deep_to_structure(dtype):
    data = [
        None,
        True,
        "foo",
        1.0,
        {False},
        torch.tensor(0.0),
        (torch.tensor(1.0), torch.tensor([2.0, 3.0])),
        [torch.tensor(4.0)],
        {"bar": torch.tensor(5.0)},
    ]
    actual = deep_to(data)
    expected = [
        None,
        True,
        "foo",
        1.0,
        {False},
        torch.tensor(0.0).to(dtype),
        (torch.tensor(1.0).to(dtype), torch.tensor([2.0, 3.0]).to(dtype)),
        [torch.tensor(4.0).to(dtype)],
        {"bar": torch.tensor(5.0).to(dtype)},
    ]
    assert_equal(actual, expected)


@pytest.mark.parametrize("shape", [None, (), (4,), (3, 2)], ids=str)
def test_deep_to_jit(shape):
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    data = torch.randn(5, 1, 1)

    def fn(loc, scale, data):
        d = dist.Normal(loc, scale, validate_args=False)
        if shape is not None:
            d = d.expand(shape)
        d = deep_to(d)
        # do some arbitrary math
        y = (data - d.rsample()) / 2
        return d.log_prob(y) + d.entropy()

    jit_fn = torch.jit.trace(fn, (loc, scale, data), check_trace=False)

    # Check value.
    torch.manual_seed(12345)
    expected = fn(loc, scale, data)
    torch.manual_seed(12345)
    actual = jit_fn(loc, scale, data)
    assert_equal(actual, expected)

    # Check grads.
    actual_grads = grad(actual.sum(), [loc, scale], retain_graph=True)
    expected_grads = grad(actual.sum(), [loc, scale])
    for a, e in zip(actual_grads, expected_grads):
        assert_equal(a, e)


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
def test_deep_to_module(dtype):
    class ExampleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("b", torch.zeros(3))
            self.c = torch.randn(4)  # this wouldn't work with torch.nn.Module.to()
            self.d = dist.Normal(0, 1)

    module = ExampleModule()
    module = deep_to(module, dtype=dtype)
    assert isinstance(module, ExampleModule)
    assert module.a.dtype == dtype
    assert module.b.dtype == dtype
    assert module.c.dtype == dtype
    assert module.d.loc.dtype == dtype
    assert module.d.scale.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float, torch.double], ids=str)
def test_deep_to_pyro_module(dtype):
    class ExampleModule(PyroModule):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("b", torch.zeros(3))
            self.c = torch.randn(4)  # this wouldn't work with torch.nn.Module.to()
            self.d = dist.Normal(0, 1)
            self.e = PyroParam(
                torch.randn(()),
                constraint=constraints.greater_than(torch.tensor(0.5)),
            )
            self.f = PyroSample(dist.Normal(0, 1))
            self.g = PyroSample(lambda self: dist.Normal(self.f, 1))

    module = ExampleModule()
    module = deep_to(module, dtype=dtype)
    assert isinstance(module, ExampleModule)
    assert module.a.dtype == dtype
    assert module.b.dtype == dtype
    assert module.c.dtype == dtype
    assert module.d.loc.dtype == dtype
    assert module.d.scale.dtype == dtype
    assert module._pyro_params["e"][0].lower_bound.dtype == dtype
    assert module.e.dtype == dtype
    assert module.f.dtype == dtype
    assert module.g.dtype == dtype
