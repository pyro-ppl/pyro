# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import config_enumerate

from pyro.contrib.funsor.enum_messenger import EnumMessenger, PlateMessenger

logger = logging.getLogger(__name__)

# These tests are currently not functioning due to missing stuff in contrib.funsor


def pyro_plate(*args, **kwargs):
    return PlateMessenger(*args, **kwargs)


def assert_ok(model, max_plate_nesting=None, **kwargs):
    """
    Assert that enumeration runs...
    """
    with poutine.enum(first_available_dim=-max_plate_nesting - 1):
        model(**kwargs)
    with EnumMessenger(first_available_dim=-max_plate_nesting - 1):
        model(**kwargs)


@pytest.mark.parametrize('enumerate_,expand', [
    (None, False),
    ("sequential", False),
    ("sequential", True),
    ("parallel", False),
    ("parallel", True),
])
def test_enumerate_parallel_plate_ok(enumerate_, expand):

    def model():
        p2 = torch.ones(2) / 2
        p34 = torch.ones(3, 4) / 4
        p536 = torch.ones(5, 3, 6) / 6

        x2 = pyro.sample("x2", dist.Categorical(p2))
        with pyro_plate("outer", 3):
            x34 = pyro.sample("x34", dist.Categorical(p34))
            with pyro_plate("inner", 5):
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
        inner_plate = pyro_plate("plate", 10)
        for i in pyro.plate("iplate", 3):  # use regular sequential plate here...
            pyro.sample("y_{}".format(i), dist.Bernoulli(0.5))
            with inner_plate:
                pyro.sample("x_{}".format(i), dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})

    assert_ok(model, max_plate_nesting=max_plate_nesting)


@pytest.mark.parametrize('enumerate_,expand', [
    (None, True),
    ("parallel", True),
    ("parallel", False),
])
def test_enum_discrete_plate_shape_broadcasting_ok(enumerate_, expand):

    def model():
        x_plate = pyro_plate("x_plate", 5, dim=-1)
        y_plate = pyro_plate("y_plate", 6, dim=-2)
        with pyro_plate("num_particles", 50, dim=-3):
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
