# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch

from pyro.ops.indexing import Vindex
from tests.common import xfail_param

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor

    import pyro.contrib.funsor

    funsor.set_backend("torch")
    from pyroapi import distributions as dist
    from pyroapi import infer, pyro

    from tests.contrib.funsor.test_valid_models_enum import assert_ok
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("enumerate_", [None, "parallel", "sequential"])
def test_enum_discrete_non_enumerated_plate_ok(enumerate_):
    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={"enumerate": "parallel"})

        with pyro.plate("non_enum", 2):
            a = pyro.sample("a", dist.Bernoulli(0.5), infer={"enumerate": None})

        p = (1.0 + a.sum(-1)) / (2.0 + a.shape[0])  # introduce dependency of b on a

        with pyro.plate("enum_1", 3):
            pyro.sample("b", dist.Bernoulli(p), infer={"enumerate": enumerate_})

    assert_ok(model, max_plate_nesting=1)


@pytest.mark.parametrize(
    "plate_dims",
    [
        (None, None, None, None),
        (-3, None, None, None),
        (None, -3, None, None),
        (-2, -3, None, None),
    ],
)
def test_plate_dim_allocation_ok(plate_dims):
    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate("plate_outer", 5, dim=plate_dims[0]):
            pyro.sample("x", dist.Bernoulli(p))
            with pyro.plate("plate_inner_1", 6, dim=plate_dims[1]):
                pyro.sample("y", dist.Bernoulli(p))
                with pyro.plate("plate_inner_2", 7, dim=plate_dims[2]):
                    pyro.sample("z", dist.Bernoulli(p))
                    with pyro.plate("plate_inner_3", 8, dim=plate_dims[3]):
                        pyro.sample("q", dist.Bernoulli(p))

    assert_ok(model, max_plate_nesting=4)


@pytest.mark.parametrize(
    "tmc_strategy",
    [None, xfail_param("diagonal", reason="strategy not implemented yet")],
)
@pytest.mark.parametrize("subsampling", [False, True])
@pytest.mark.parametrize("reuse_plate", [False, True])
def test_enum_recycling_plate(subsampling, reuse_plate, tmc_strategy):
    @infer.config_enumerate(
        default="parallel", tmc=tmc_strategy, num_samples=2 if tmc_strategy else None
    )
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        q = pyro.param("q", torch.tensor([0.5, 0.5]))
        plate_x = pyro.plate(
            "plate_x", 4, subsample_size=3 if subsampling else None, dim=-1
        )
        plate_y = pyro.plate(
            "plate_y", 5, subsample_size=3 if subsampling else None, dim=-1
        )
        plate_z = pyro.plate(
            "plate_z", 6, subsample_size=3 if subsampling else None, dim=-2
        )

        a = pyro.sample("a", dist.Bernoulli(q[0])).long()
        w = 0
        for i in pyro.markov(range(4)):
            w = pyro.sample("w_{}".format(i), dist.Categorical(p[w]))

        with plate_x:
            b = pyro.sample("b", dist.Bernoulli(q[a])).long()
            x = 0
            for i in pyro.markov(range(4)):
                x = pyro.sample("x_{}".format(i), dist.Categorical(p[x]))

        with plate_y:
            c = pyro.sample("c", dist.Bernoulli(q[a])).long()
            y = 0
            for i in pyro.markov(range(4)):
                y = pyro.sample("y_{}".format(i), dist.Categorical(p[y]))

        with plate_z:
            d = pyro.sample("d", dist.Bernoulli(q[a])).long()
            z = 0
            for i in pyro.markov(range(4)):
                z = pyro.sample("z_{}".format(i), dist.Categorical(p[z]))

        with plate_x, plate_z:
            # this part is tricky: how do we know to preserve b's dimension?
            # also, how do we know how to make b and d have different dimensions?
            e = pyro.sample("e", dist.Bernoulli(q[b if reuse_plate else a])).long()
            xz = 0
            for i in pyro.markov(range(4)):
                xz = pyro.sample("xz_{}".format(i), dist.Categorical(p[xz]))

        return a, b, c, d, e

    assert_ok(model, max_plate_nesting=2)


@pytest.mark.parametrize("enumerate_", [None, "parallel", "sequential"])
@pytest.mark.parametrize("reuse_plate", [True, False])
def test_enum_discrete_plates_dependency_ok(enumerate_, reuse_plate):
    @infer.config_enumerate(default=enumerate_)
    def model():
        x_plate = pyro.plate("x_plate", 10, dim=-1)
        y_plate = pyro.plate("y_plate", 11, dim=-2)
        q = pyro.param("q", torch.tensor([0.5, 0.5]))
        pyro.sample("a", dist.Bernoulli(0.5))
        with x_plate:
            b = pyro.sample("b", dist.Bernoulli(0.5)).long()
        with y_plate:
            # Note that it is difficult to check that c does not depend on b.
            c = pyro.sample("c", dist.Bernoulli(0.5)).long()
        with x_plate, y_plate:
            pyro.sample("d", dist.Bernoulli(Vindex(q)[b] if reuse_plate else 0.5))

        assert c.shape != b.shape or enumerate_ == "sequential"

    assert_ok(model, max_plate_nesting=2)


@pytest.mark.parametrize("subsampling", [False, True])
@pytest.mark.parametrize("enumerate_", [None, "parallel", "sequential"])
def test_enum_discrete_plate_shape_broadcasting_ok(subsampling, enumerate_):
    @infer.config_enumerate(default=enumerate_)
    def model():
        x_plate = pyro.plate(
            "x_plate", 5, subsample_size=2 if subsampling else None, dim=-1
        )
        y_plate = pyro.plate(
            "y_plate", 6, subsample_size=3 if subsampling else None, dim=-2
        )
        with pyro.plate("num_particles", 50, dim=-3):
            with x_plate:
                b = pyro.sample("b", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            with y_plate:
                c = pyro.sample("c", dist.Bernoulli(0.5))
            with x_plate, y_plate:
                d = pyro.sample("d", dist.Bernoulli(b))

        # check shapes
        if enumerate_ == "parallel":
            assert b.shape == (50, 1, x_plate.subsample_size)
            assert c.shape == (2, 1, 1, 1)
            assert d.shape == (2, 1, 1, 1, 1)
        elif enumerate_ == "sequential":
            assert b.shape == (50, 1, x_plate.subsample_size)
            assert c.shape in ((), (1, 1, 1))  # both are valid
            assert d.shape in ((), (1, 1, 1))  # both are valid
        else:
            assert b.shape == (50, 1, x_plate.subsample_size)
            assert c.shape == (50, y_plate.subsample_size, 1)
            assert d.shape == (50, y_plate.subsample_size, x_plate.subsample_size)

    assert_ok(model, guide=model, max_plate_nesting=3)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("num_samples", [None, 2])
def test_plate_subsample_primitive_ok(subsample_size, num_samples):
    @infer.config_enumerate(num_samples=num_samples, tmc="full")
    def model():
        with pyro.plate("plate", 10, subsample_size=subsample_size, dim=None):
            p0 = torch.tensor(0.0)
            p0 = pyro.subsample(p0, event_dim=0)
            assert p0.shape == ()
            p = 0.5 * torch.ones(10)
            p = pyro.subsample(p, event_dim=0)
            assert len(p) == (subsample_size if subsample_size else 10)
            pyro.sample("x", dist.Bernoulli(p))

    assert_ok(model, max_plate_nesting=1)
