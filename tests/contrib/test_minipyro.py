# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
import torch
from torch.distributions import constraints

from pyro.generic import distributions as dist
from pyro.generic import infer, ops, optim, pyro, pyro_backend
from tests.common import assert_close, xfail_param

# This file tests a variety of model,guide pairs with valid and invalid structure.
# See https://github.com/pyro-ppl/pyro/blob/0.3.1/tests/infer/test_valid_models.py


def assert_ok(model, guide, elbo, *args, **kwargs):
    """
    Assert that inference works without warnings or errors.
    """
    pyro.get_param_store().clear()
    adam = optim.Adam({"lr": 1e-6})
    inference = infer.SVI(model, guide, adam, elbo)
    for i in range(2):
        inference.step(*args, **kwargs)


def assert_error(model, guide, elbo, match=None):
    """
    Assert that inference fails with an error.
    """
    pyro.get_param_store().clear()
    adam = optim.Adam({"lr": 1e-6})
    inference = infer.SVI(model,  guide, adam, elbo)
    with pytest.raises((NotImplementedError, UserWarning, KeyError, ValueError, RuntimeError),
                       match=match):
        inference.step()


def assert_warning(model, guide, elbo):
    """
    Assert that inference works but with a warning.
    """
    pyro.get_param_store().clear()
    adam = optim.Adam({"lr": 1e-6})
    inference = infer.SVI(model, guide, adam, elbo)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        inference.step()
        assert len(w), 'No warnings were raised'
        for warning in w:
            print(warning)


@pytest.mark.parametrize("backend", ["pyro", "minipyro"])
def test_generate_data(backend):

    def model(data=None):
        loc = pyro.param("loc", torch.tensor(2.0))
        scale = pyro.param("scale", torch.tensor(1.0))
        x = pyro.sample("x", dist.Normal(loc, scale), obs=data)
        return x

    with pyro_backend(backend):
        data = model().data
        assert data.shape == ()


@pytest.mark.parametrize("backend", ["pyro", "minipyro"])
def test_generate_data_plate(backend):
    num_points = 1000

    def model(data=None):
        loc = pyro.param("loc", torch.tensor(2.0))
        scale = pyro.param("scale", torch.tensor(1.0))
        with pyro.plate("data", 1000, dim=-1):
            x = pyro.sample("x", dist.Normal(loc, scale), obs=data)
        return x

    with pyro_backend(backend):
        data = model().data
        assert data.shape == (num_points,)
        mean = float(ops.sum(data)) / num_points
        assert 1.9 <= mean <= 2.1


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", ["pyro", "minipyro"])
def test_nonempty_model_empty_guide_ok(backend, jit):

    def model(data):
        loc = pyro.param("loc", torch.tensor(0.0))
        pyro.sample("x", dist.Normal(loc, 1.), obs=data)

    def guide(data):
        pass

    data = torch.tensor(2.)
    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo, data)


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", ["pyro", "minipyro"])
def test_plate_ok(backend, jit):
    data = torch.randn(10)

    def model():
        locs = pyro.param("locs", torch.tensor([0.2, 0.3, 0.5]))
        p = torch.tensor([0.2, 0.3, 0.5])
        with pyro.plate("plate", len(data), dim=-1):
            x = pyro.sample("x", dist.Categorical(p))
            pyro.sample("obs", dist.Normal(locs[x], 1.), obs=data)

    def guide():
        p = pyro.param("p", torch.tensor([0.5, 0.3, 0.2]))
        with pyro.plate("plate", len(data), dim=-1):
            pyro.sample("x", dist.Categorical(p))

    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", ["pyro", "minipyro"])
def test_nested_plate_plate_ok(backend, jit):
    data = torch.randn(2, 3)

    def model():
        loc = torch.tensor(3.0)
        with pyro.plate("plate_outer", data.size(-1), dim=-1):
            x = pyro.sample("x", dist.Normal(loc, 1.))
            with pyro.plate("plate_inner", data.size(-2), dim=-2):
                pyro.sample("y", dist.Normal(x, 1.), obs=data)

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        scale = pyro.param("scale", torch.tensor(1.))
        with pyro.plate("plate_outer", data.size(-1), dim=-1):
            pyro.sample("x", dist.Normal(loc, scale))

    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", [
    "pyro",
    xfail_param("minipyro", reason="not implemented"),
])
def test_local_param_ok(backend, jit):
    data = torch.randn(10)

    def model():
        locs = pyro.param("locs", torch.tensor([-1., 0., 1.]))
        with pyro.plate("plate", len(data), dim=-1):
            x = pyro.sample("x", dist.Categorical(torch.ones(3) / 3))
            pyro.sample("obs", dist.Normal(locs[x], 1.), obs=data)

    def guide():
        with pyro.plate("plate", len(data), dim=-1):
            p = pyro.param("p", torch.ones(len(data), 3) / 3, event_dim=1)
            pyro.sample("x", dist.Categorical(p))
        return p

    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)

        # Check that pyro.param() can be called without init_value.
        expected = guide()
        actual = pyro.param("p")
        assert_close(actual, expected)


@pytest.mark.parametrize("jit", [False, True], ids=["py", "jit"])
@pytest.mark.parametrize("backend", ["pyro", "minipyro"])
def test_constraints(backend, jit):
    data = torch.tensor(0.5)

    def model():
        locs = pyro.param("locs", torch.randn(3), constraint=constraints.real)
        scales = pyro.param("scales", ops.exp(torch.randn(3)), constraint=constraints.positive)
        p = torch.tensor([0.5, 0.3, 0.2])
        x = pyro.sample("x", dist.Categorical(p))
        pyro.sample("obs", dist.Normal(locs[x], scales[x]), obs=data)

    def guide():
        q = pyro.param("q", ops.exp(torch.randn(3)), constraint=constraints.simplex)
        pyro.sample("x", dist.Categorical(q))

    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)
