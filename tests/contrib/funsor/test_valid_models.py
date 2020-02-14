# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from collections import defaultdict

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import config_enumerate
from pyro.infer.util import torch_item
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from tests.common import assert_close

from pyro.contrib.funsor.enum_messenger import EnumMessenger as FunsorEnumMessenger

logger = logging.getLogger(__name__)


def assert_ok(model, max_plate_nesting=None, **kwargs):
    """
    Assert that enumeration runs...
    """
    pyro.clear_param_store()
    with poutine.enum(first_available_dim=-max_plate_nesting - 1):
        model(**kwargs)
    with FunsorEnumMessenger(first_available_dim=-max_plate_nesting - 1):
        model(**kwargs)


@pytest.mark.parametrize('enumerate_,expand,num_samples', [
    (None, False, None),
    ("sequential", False, None),
    ("sequential", True, None),
    ("parallel", False, None),
    ("parallel", True, None),
])
def test_enumerate_parallel_plate_ok(enumerate_, expand, num_samples):

    def model():
        p2 = torch.ones(2) / 2
        p34 = torch.ones(3, 4) / 4
        p536 = torch.ones(5, 3, 6) / 6

        x2 = pyro.sample("x2", dist.Categorical(p2))
        with pyro.plate("outer", 3):
            x34 = pyro.sample("x34", dist.Categorical(p34))
            with pyro.plate("inner", 5):
                x536 = pyro.sample("x536", dist.Categorical(p536))

        # check shapes
        if enumerate_ == "parallel":
            if expand:
                # Meaning of dimensions:    [ enum dims | plate dims ]
                assert x2.shape == torch.Size([        2, 1, 1])  # noqa: E201
                assert x34.shape == torch.Size([    4, 1, 1, 3])  # noqa: E201
                assert x536.shape == torch.Size([6, 1, 1, 5, 3])  # noqa: E201
            else:
                # Meaning of dimensions:    [ enum dims | plate placeholders ]
                assert x2.shape == torch.Size([        2, 1, 1])  # noqa: E201
                assert x34.shape == torch.Size([    4, 1, 1, 1])  # noqa: E201
                assert x536.shape == torch.Size([6, 1, 1, 1, 1])  # noqa: E201
        elif enumerate_ == "sequential":
            if expand:
                # All dimensions are plate dimensions.
                assert x2.shape == torch.Size([])
                assert x34.shape == torch.Size([3])
                assert x536.shape == torch.Size([5, 3])
            else:
                # All dimensions are plate placeholders.
                assert x2.shape == torch.Size([])
                assert x34.shape == torch.Size([1])
                assert x536.shape == torch.Size([1, 1])
        else:
            # All dimensions are plate dimensions.
            assert x2.shape == torch.Size([])
            assert x34.shape == torch.Size([3])
            assert x536.shape == torch.Size([5, 3])

    model = config_enumerate(model, enumerate_, expand)
    assert_ok(model, max_plate_nesting=2)


@pytest.mark.parametrize('max_plate_nesting', [1, float('inf')])
@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
def test_enum_discrete_iplate_plate_dependency_ok(enumerate_, max_plate_nesting):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_plate = pyro.plate("plate", 10, 5)
        for i in pyro.plate("iplate", 3):
            pyro.sample("y_{}".format(i), dist.Bernoulli(0.5))
            with inner_plate:
                pyro.sample("x_{}".format(i), dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})

    assert_ok(model, max_plate_nesting=max_plate_nesting)


@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
def test_enum_discrete_plates_dependency_ok(enumerate_):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        x_plate = pyro.plate("x_plate", 10, 5, dim=-1)
        y_plate = pyro.plate("y_plate", 11, 6, dim=-2)
        pyro.sample("a", dist.Bernoulli(0.5))
        with x_plate:
            pyro.sample("b", dist.Bernoulli(0.5).expand_by([5]))
        with y_plate:
            # Note that it is difficult to check that c does not depend on b.
            pyro.sample("c", dist.Bernoulli(0.5).expand_by([6, 1]))
        with x_plate, y_plate:
            pyro.sample("d", dist.Bernoulli(0.5).expand_by([6, 5]))

    assert_ok(model, max_plate_nesting=2)


@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
def test_enum_discrete_non_enumerated_plate_ok(enumerate_):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})

        with pyro.plate("non_enum", 2):
            a = pyro.sample("a", dist.Bernoulli(0.5).expand_by([2]),
                            infer={'enumerate': None})

        p = (1.0 + a.sum(-1)) / (2.0 + a.size(0))  # introduce dependency of b on a

        with pyro.plate("enum_1", 3):
            pyro.sample("b", dist.Bernoulli(p).expand_by([3]),
                        infer={'enumerate': enumerate_})

    assert_ok(model, max_plate_nesting=1)


def test_plate_shape_broadcasting():
    data = torch.ones(1000, 2)

    def model():
        with pyro.plate("num_particles", 10, dim=-3):
            with pyro.plate("components", 2, dim=-1):
                p = pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
                assert p.shape == torch.Size((10, 1, 2))
            with pyro.plate("data", data.shape[0], dim=-2):
                pyro.sample("obs", dist.Bernoulli(p), obs=data)

    assert_ok(model)


@pytest.mark.parametrize('enumerate_,expand,num_samples', [
    (None, True, None),
    ("parallel", True, None),
    ("parallel", False, None),
])
def test_enum_discrete_plate_shape_broadcasting_ok(enumerate_, expand, num_samples):

    def model():
        x_plate = pyro.plate("x_plate", 10, 5, dim=-1)
        y_plate = pyro.plate("y_plate", 11, 6, dim=-2)
        with pyro.plate("num_particles", 50, dim=-3):
            with x_plate:
                b = pyro.sample("b", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            with y_plate:
                c = pyro.sample("c", dist.Bernoulli(0.5))
            with x_plate, y_plate:
                d = pyro.sample("d", dist.Bernoulli(b))

        # check shapes
        if enumerate_ == "parallel":
            if expand:
                assert b.shape == (50, 1, 5)
                assert c.shape == (2, 50, 6, 1)
                assert d.shape == (2, 1, 50, 6, 5)
            else:
                assert b.shape == (50, 1, 5)
                assert c.shape == (2, 1, 1, 1)
                assert d.shape == (2, 1, 1, 1, 1)
        else:
            assert b.shape == (50, 1, 5)
            assert c.shape == (50, 6, 1)
            assert d.shape == (50, 6, 5)

    model = config_enumerate(model, default=enumerate_, expand=expand)
    assert_ok(model, max_plate_nesting=3)


@pytest.mark.parametrize("expand", [True, False])
def test_dim_allocation_ok(expand):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate("plate_outer", 10, 5, dim=-3):
            x = pyro.sample("x", dist.Bernoulli(p))
            with pyro.plate("plate_inner_1", 11, 6):
                y = pyro.sample("y", dist.Bernoulli(p))
                # allocated dim is rightmost available, i.e. -1
                with pyro.plate("plate_inner_2", 12, 7):
                    z = pyro.sample("z", dist.Bernoulli(p))
                    # allocated dim is next rightmost available, i.e. -2
                    # since dim -3 is already allocated, use dim=-4
                    with pyro.plate("plate_inner_3", 13, 8):
                        q = pyro.sample("q", dist.Bernoulli(p))

        # check shapes
        if enumerate_ and not expand:
            assert x.shape == (1, 1, 1)
            assert y.shape == (1, 1, 1)
            assert z.shape == (1, 1, 1)
            assert q.shape == (1, 1, 1, 1)
        else:
            assert x.shape == (5, 1, 1)
            assert y.shape == (5, 1, 6)
            assert z.shape == (5, 7, 6)
            assert q.shape == (8, 5, 7, 6)

    assert_ok(model, max_plate_nesting=4)
