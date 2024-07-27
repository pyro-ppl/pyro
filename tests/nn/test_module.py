# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import io
import math
import warnings
from typing import Callable, Iterable

import pytest
import torch
from torch import nn
from torch.distributions import constraints, transform_to

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoDiagonalNormal
from pyro.nn.module import PyroModule, PyroParam, PyroSample, clear, to_pyro_module_
from pyro.optim import Adam
from tests.common import assert_equal, xfail_param


def test_svi_smoke():
    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.loc = nn.Parameter(torch.zeros(2))
            self.scale = PyroParam(torch.ones(2), constraint=constraints.positive)
            self.z = PyroSample(
                lambda self: dist.Normal(self.loc, self.scale).to_event(1)
            )

        def forward(self, data):
            loc, log_scale = self.z.unbind(-1)
            with pyro.plate("data"):
                pyro.sample("obs", dist.Cauchy(loc, log_scale.exp()), obs=data)

    class Guide(PyroModule):
        def __init__(self):
            super().__init__()
            self.loc = nn.Parameter(torch.zeros(2))
            self.scale = PyroParam(torch.ones(2), constraint=constraints.positive)
            self.z = PyroSample(
                lambda self: dist.Normal(self.loc, self.scale).to_event(1)
            )

        def forward(self, *args, **kwargs):
            return self.z

    data = torch.randn(5)
    model = Model()
    trace = poutine.trace(model).get_trace(data)
    assert "loc" in trace.nodes.keys()
    assert trace.nodes["loc"]["type"] == "param"
    assert "scale" in trace.nodes
    assert trace.nodes["scale"]["type"] == "param"

    guide = Guide()
    trace = poutine.trace(guide).get_trace(data)
    assert "loc" in trace.nodes.keys()
    assert trace.nodes["loc"]["type"] == "param"
    assert "scale" in trace.nodes
    assert trace.nodes["scale"]["type"] == "param"

    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, Trace_ELBO())
    for step in range(3):
        svi.step(data)


@pytest.mark.parametrize("local_params", [True, False])
@pytest.mark.parametrize("num_particles", [1, 2])
@pytest.mark.parametrize("vectorize_particles", [True, False])
@pytest.mark.parametrize("Autoguide", [pyro.infer.autoguide.AutoNormal])
def test_svi_elbomodule_interface(
    local_params, num_particles, vectorize_particles, Autoguide
):
    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.loc = nn.Parameter(torch.zeros(2))
            self.scale = PyroParam(torch.ones(2), constraint=constraints.positive)
            self.z = PyroSample(
                lambda self: dist.Normal(self.loc, self.scale).to_event(1)
            )

        def forward(self, data):
            loc, log_scale = self.z.unbind(-1)
            with pyro.plate("data"):
                pyro.sample("obs", dist.Cauchy(loc, log_scale.exp()), obs=data)

    with pyro.settings.context(module_local_params=local_params):
        data = torch.randn(5)
        model = Model()
        model(data)  # initialize

        guide = Autoguide(model)
        guide(data)  # initialize

        elbo = Trace_ELBO(
            vectorize_particles=vectorize_particles, num_particles=num_particles
        )
        elbo = elbo(model, guide)
        assert isinstance(elbo, torch.nn.Module)
        assert set(
            k[: -len("_unconstrained")] if k.endswith("_unconstrained") else k
            for k, v in elbo.named_parameters()
        ) == set("model." + k for k, v in model.named_pyro_params()) | set(
            "guide." + k for k, v in guide.named_pyro_params()
        )

        adam = torch.optim.Adam(elbo.parameters(), lr=0.0001)
        for _ in range(3):
            adam.zero_grad()
            loss = elbo(data)
            loss.backward()
            adam.step()

        guide2 = Autoguide(model)
        guide2(data)  # initialize
        if local_params:
            assert set(pyro.get_param_store().keys()) == set()
            for (name, p), (name2, p2) in zip(
                guide.named_parameters(), guide2.named_parameters()
            ):
                assert name == name2
                assert not torch.allclose(p, p2)
        else:
            assert set(pyro.get_param_store().keys()) != set()
            for (name, p), (name2, p2) in zip(
                guide.named_parameters(), guide2.named_parameters()
            ):
                assert name == name2
                assert torch.allclose(p, p2)


@pytest.mark.parametrize("local_params", [True, False])
def test_local_param_global_behavior_fails(local_params):
    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.global_nn_param = nn.Parameter(torch.zeros(2))

        def forward(self):
            global_param = pyro.param("_global_param", lambda: torch.randn(2))
            global_nn_param = pyro.param("global_nn_param", self.global_nn_param)
            return global_param, global_nn_param

    with pyro.settings.context(module_local_params=local_params):
        model = Model()
        if local_params:
            assert pyro.settings.get("module_local_params")
            with pytest.raises(NotImplementedError):
                model()
        else:
            assert not pyro.settings.get("module_local_params")
            model()


@pytest.mark.parametrize("local_params", [True, False])
def test_names(local_params):
    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor(0.0))
            self.y = PyroParam(torch.tensor(1.0), constraint=constraints.positive)
            self.m = nn.Module()
            self.m.u = nn.Parameter(torch.tensor(2.0))
            self.p = PyroModule()
            self.p.v = nn.Parameter(torch.tensor(3.0))
            self.p.w = PyroParam(torch.tensor(4.0), constraint=constraints.positive)

        def forward(self):
            # trigger .__getattr__()
            self.x
            self.y
            self.m
            self.p.v
            self.p.w

    with pyro.settings.context(module_local_params=local_params):
        model = Model()

        # Check named_parameters.
        expected = {
            "x",
            "y_unconstrained",
            "m.u",
            "p.v",
            "p.w_unconstrained",
        }
        actual = set(name for name, _ in model.named_parameters())
        assert actual == expected

        # Check pyro.param names.
        expected = {"x", "y", "m$$$u", "p.v", "p.w"}
        with poutine.trace(param_only=True) as param_capture:
            model()
        actual = {
            name
            for name, site in param_capture.trace.nodes.items()
            if site["type"] == "param"
        }
        assert actual == expected
        if local_params:
            assert set(pyro.get_param_store().keys()) == set()
        else:
            assert set(pyro.get_param_store().keys()) == expected

        # Check pyro_parameters method
        expected = {"x", "y", "m.u", "p.v", "p.w"}
        actual = set(k for k, v in model.named_pyro_params())
        assert actual == expected


def test_delete():
    m = PyroModule()
    m.a = PyroParam(torch.tensor(1.0))
    del m.a
    m.a = PyroParam(torch.tensor(0.1))
    assert_equal(m.a.detach(), torch.tensor(0.1))


def test_nested():
    class Child(PyroModule):
        def __init__(self, a):
            super().__init__()
            self.a = PyroParam(a, constraints.positive)

    class Family(PyroModule):
        def __init__(self):
            super().__init__()
            self.child1 = Child(torch.tensor(1.0))
            self.child2 = Child(torch.tensor(2.0))

    f = Family()
    assert_equal(f.child1.a.detach(), torch.tensor(1.0))
    assert_equal(f.child2.a.detach(), torch.tensor(2.0))


def test_module_cache():
    class Child(PyroModule):
        def __init__(self, x):
            super().__init__()
            self.a = PyroParam(torch.tensor(x))

        def forward(self):
            return self.a

    class Family(PyroModule):
        def __init__(self):
            super().__init__()
            self.c = Child(1.0)

        def forward(self):
            return self.c.a

    f = Family()
    assert_equal(f().detach(), torch.tensor(1.0))
    f.c = Child(3.0)
    assert_equal(f().detach(), torch.tensor(3.0))
    assert_equal(f.c().detach(), torch.tensor(3.0))


def test_submodule_contains_torch_module():
    submodule = PyroModule()
    submodule.linear = nn.Linear(1, 1)
    module = PyroModule()
    module.child = submodule


def test_hierarchy_prior_cached():
    def hierarchy_prior(module):
        latent = pyro.sample("a", dist.Normal(0, 1))
        return dist.Normal(latent, 1)

    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.b = PyroSample(hierarchy_prior)

        def forward(self):
            return self.b + self.b

    model = Model()
    trace = poutine.trace(model).get_trace()
    assert "a" in trace.nodes


SHAPE_CONSTRAINT = [
    ((), constraints.real),
    ((4,), constraints.real),
    ((3, 2), constraints.real),
    ((), constraints.positive),
    ((4,), constraints.positive),
    ((3, 2), constraints.positive),
    ((5,), constraints.simplex),
    (
        (
            2,
            5,
        ),
        constraints.simplex,
    ),
    ((5, 5), constraints.lower_cholesky),
    ((2, 5, 5), constraints.lower_cholesky),
    ((10,), constraints.greater_than(-torch.randn(10).exp())),
    ((4, 10), constraints.greater_than(-torch.randn(10).exp())),
    ((4, 10), constraints.greater_than(-torch.randn(4, 10).exp())),
    ((3, 2, 10), constraints.greater_than(-torch.randn(10).exp())),
    ((3, 2, 10), constraints.greater_than(-torch.randn(2, 10).exp())),
    ((3, 2, 10), constraints.greater_than(-torch.randn(3, 1, 10).exp())),
    ((3, 2, 10), constraints.greater_than(-torch.randn(3, 2, 10).exp())),
    ((5,), constraints.real_vector),
    (
        (
            2,
            5,
        ),
        constraints.real_vector,
    ),
    ((), constraints.unit_interval),
    ((4,), constraints.unit_interval),
    ((3, 2), constraints.unit_interval),
    ((10,), constraints.interval(-torch.randn(10).exp(), torch.randn(10).exp())),
    ((4, 10), constraints.interval(-torch.randn(10).exp(), torch.randn(10).exp())),
    ((3, 2, 10), constraints.interval(-torch.randn(10).exp(), torch.randn(10).exp())),
]


@pytest.mark.parametrize("shape,constraint_", SHAPE_CONSTRAINT)
def test_constraints(shape, constraint_):
    module = PyroModule()
    module.x = PyroParam(torch.full(shape, 1e-4), constraint_)

    assert isinstance(module.x, torch.Tensor)
    assert isinstance(module.x_unconstrained, nn.Parameter)
    assert module.x.shape == shape
    assert constraint_.check(module.x).all()

    module.x = torch.randn(shape).exp() * 1e-6
    assert isinstance(module.x_unconstrained, nn.Parameter)
    assert isinstance(module.x, torch.Tensor)
    assert module.x.shape == shape
    assert constraint_.check(module.x).all()

    assert isinstance(module.x_unconstrained, torch.Tensor)
    y = module.x_unconstrained.data.normal_()
    assert_equal(module.x.data, transform_to(constraint_)(y))
    assert constraint_.check(module.x).all()

    del module.x
    assert "x" not in module._pyro_params
    assert not hasattr(module, "x")
    assert not hasattr(module, "x_unconstrained")


@pytest.mark.parametrize("local_params", [True, False])
def test_clear(local_params):
    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor(0.0))
            self.m = torch.nn.Linear(2, 3)
            self.m.weight.data.fill_(1.0)
            self.m.bias.data.fill_(2.0)
            self.p = PyroModule()
            self.p.x = nn.Parameter(torch.tensor(3.0))

        def forward(self):
            return [x.clone() for x in [self.x, self.m.weight, self.m.bias, self.p.x]]

    with pyro.settings.context(module_local_params=local_params):
        m = Model()
        state0 = m()

        # mutate
        for _, x in m.named_pyro_params():
            if hasattr(x, "unconstrained"):
                x = x.unconstrained()
            x.data += torch.randn(x.shape)
        state1 = m()
        for x, y in zip(state0, state1):
            assert not (x == y).all()

        if local_params:
            assert set(pyro.get_param_store().keys()) == set()
        else:
            assert set(pyro.get_param_store().keys()) == {
                "x",
                "m$$$weight",
                "m$$$bias",
                "p.x",
            }
            clear(m)
            del m
            assert set(pyro.get_param_store().keys()) == set()

        m = Model()
        state2 = m()
        if local_params:
            assert set(pyro.get_param_store().keys()) == set()
        else:
            assert set(pyro.get_param_store().keys()) == {
                "x",
                "m$$$weight",
                "m$$$bias",
                "p.x",
            }
        for actual, expected in zip(state2, state0):
            assert_equal(actual, expected)


def test_sample():
    class Model(nn.Linear, PyroModule):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features)
            self.weight = PyroSample(
                lambda self: dist.Normal(0, 1)
                .expand([self.out_features, self.in_features])
                .to_event(2)
            )

    class Guide(nn.Linear, PyroModule):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features)
            self.loc = PyroParam(torch.zeros_like(self.weight))
            self.scale = PyroParam(
                torch.ones_like(self.weight), constraint=constraints.positive
            )
            self.weight = PyroSample(
                lambda self: dist.Normal(self.loc, self.scale).to_event(2)
            )

    data = torch.randn(8)
    model = Model(8, 2)
    guide = Guide(8, 2)

    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, Trace_ELBO())
    for step in range(3):
        svi.step(data)


def test_cache():
    class MyModule(PyroModule):
        def forward(self):
            return [self.gather(), self.gather()]

        def gather(self):
            return {
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "p.d": self.p.d,
                "p.e": self.p.e,
                "p.f": self.p.f,
            }

    module = MyModule()
    module.a = nn.Parameter(torch.tensor(0.0))
    module.b = PyroParam(torch.tensor(1.0), constraint=constraints.positive)
    module.c = PyroSample(dist.Normal(0, 1))
    module.p = PyroModule()
    module.p.d = nn.Parameter(torch.tensor(3.0))
    module.p.e = PyroParam(torch.tensor(4.0), constraint=constraints.positive)
    module.p.f = PyroSample(dist.Normal(0, 1))

    assert module._pyro_context is module.p._pyro_context

    # Check that results are cached with an invocation of .__call__().
    result1 = module()
    actual, expected = result1
    for key in ["a", "c", "p.d", "p.f"]:
        assert actual[key] is expected[key], key

    # Check that results are not cached across invocations of .__call__().
    result2 = module()
    for key in ["b", "c", "p.e", "p.f"]:
        assert result1[0] is not result2[0], key


class AttributeModel(PyroModule):
    def __init__(self, size):
        super().__init__()
        self.x = PyroParam(torch.zeros(size))
        self.y = PyroParam(lambda: torch.randn(size))
        self.z = PyroParam(
            torch.ones(size), constraint=constraints.positive, event_dim=1
        )
        self.s = PyroSample(dist.Normal(0, 1))
        self.t = PyroSample(lambda self: dist.Normal(self.s, self.z))
        self.u = PyroSample(lambda self: self.t**2)

    def forward(self):
        return self.x + self.y + self.u


class DecoratorModel(PyroModule):
    def __init__(self, size):
        super().__init__()
        self.size = size

    @PyroParam
    def x(self):
        return torch.zeros(self.size)

    @PyroParam
    def y(self):
        return torch.randn(self.size)

    @PyroParam(constraint=constraints.positive, event_dim=1)
    def z(self):
        return torch.ones(self.size)

    @PyroSample
    def s(self):
        return dist.Normal(0, 1)

    @PyroSample
    def t(self):
        return dist.Normal(self.s, self.z).to_event(1)

    @PyroSample
    def u(self):
        return self.t**2

    def forward(self):
        return self.x + self.y + self.u


@pytest.mark.parametrize("Model", [AttributeModel, DecoratorModel])
@pytest.mark.parametrize("size", [1, 2])
def test_decorator(Model, size):
    model = Model(size)
    for i in range(2):
        trace = poutine.trace(model).get_trace()
        assert set(trace.nodes.keys()) == {
            "_INPUT",
            "x",
            "y",
            "z",
            "s",
            "t",
            "u",
            "_RETURN",
        }

        assert trace.nodes["x"]["type"] == "param"
        assert trace.nodes["y"]["type"] == "param"
        assert trace.nodes["z"]["type"] == "param"
        assert trace.nodes["s"]["type"] == "sample"
        assert trace.nodes["t"]["type"] == "sample"
        assert trace.nodes["u"]["type"] == "sample"

        assert trace.nodes["x"]["value"].shape == (size,)
        assert trace.nodes["y"]["value"].shape == (size,)
        assert trace.nodes["z"]["value"].shape == (size,)
        assert trace.nodes["s"]["value"].shape == ()
        assert trace.nodes["t"]["value"].shape == (size,)
        assert trace.nodes["u"]["value"].shape == (size,)

        assert trace.nodes["u"]["infer"] == {"_deterministic": True}


def test_mixin_factory():
    assert PyroModule[nn.Module] is PyroModule
    assert PyroModule[PyroModule] is PyroModule

    module = PyroModule[nn.Sequential](
        PyroModule[nn.Linear](28 * 28, 200),
        PyroModule[nn.Sigmoid](),
        PyroModule[nn.Linear](200, 200),
        PyroModule[nn.Sigmoid](),
        PyroModule[nn.Linear](200, 10),
    )

    assert isinstance(module, nn.Sequential)
    assert isinstance(module, PyroModule)
    assert type(module).__name__ == "PyroSequential"
    assert PyroModule[type(module)] is type(module)

    assert isinstance(module[0], nn.Linear)
    assert isinstance(module[0], PyroModule)
    assert type(module[0]).__name__ == "PyroLinear"
    assert type(module[2]) is type(module[0])  # noqa: E721
    assert module[0]._pyro_name == "0"
    assert module[1]._pyro_name == "1"

    # Ensure new types are serializable.
    data = torch.randn(28 * 28)
    expected = module(data)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        f = io.BytesIO()
        torch.save(module, f)
        del module
        pyro.clear_param_store()
        f.seek(0)
        module = torch.load(f, weights_only=False)
    assert type(module).__name__ == "PyroSequential"
    actual = module(data)
    assert_equal(actual, expected)


def test_to_pyro_module_():
    pyro.set_rng_seed(123)
    actual = nn.Sequential(
        nn.Linear(28 * 28, 200),
        nn.Sigmoid(),
        nn.Linear(200, 200),
        nn.Sigmoid(),
        nn.Linear(200, 10),
    )
    to_pyro_module_(actual)
    pyro.clear_param_store()

    pyro.set_rng_seed(123)
    expected = PyroModule[nn.Sequential](
        PyroModule[nn.Linear](28 * 28, 200),
        PyroModule[nn.Sigmoid](),
        PyroModule[nn.Linear](200, 200),
        PyroModule[nn.Sigmoid](),
        PyroModule[nn.Linear](200, 10),
    )
    pyro.clear_param_store()

    def assert_identical(a, e):
        assert type(a) is type(e)
        if isinstance(a, dict):
            assert set(a) == set(e)
            for key in a:
                assert_identical(a[key], e[key])
        elif isinstance(a, nn.Module):
            assert_identical(a.__dict__, e.__dict__)
        elif isinstance(a, (str, int, float, torch.Tensor)):
            assert_equal(a, e)

    assert_identical(actual, expected)

    # check output
    data = torch.randn(28 * 28)
    actual_out = actual(data)
    pyro.clear_param_store()
    expected_out = expected(data)
    assert_equal(actual_out, expected_out)

    # check randomization
    def randomize(model):
        for m in model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(
                    m,
                    name,
                    PyroSample(
                        prior=dist.Normal(0, 1)
                        .expand(value.shape)
                        .to_event(value.dim())
                    ),
                )

    randomize(actual)
    randomize(expected)
    assert_identical(actual, expected)


@pytest.mark.parametrize("local_params", [True, False])
def test_torch_serialize_attributes(local_params):
    with pyro.settings.context(module_local_params=local_params):
        module = PyroModule()
        module.x = PyroParam(torch.tensor(1.234), constraints.positive)
        module.y = nn.Parameter(torch.randn(3))
        assert isinstance(module.x, torch.Tensor)

        # Work around https://github.com/pytorch/pytorch/issues/27972
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            f = io.BytesIO()
            torch.save(module, f)
            pyro.clear_param_store()
            f.seek(0)
            actual = torch.load(f, weights_only=False)

        assert_equal(actual.x, module.x)
        actual_names = {name for name, _ in actual.named_parameters()}
        expected_names = {name for name, _ in module.named_parameters()}
        assert actual_names == expected_names


@pytest.mark.parametrize("local_params", [True, False])
def test_torch_serialize_decorators(local_params):
    with pyro.settings.context(module_local_params=local_params):
        module = DecoratorModel(3)
        module()  # initialize

        module2 = DecoratorModel(3)
        module2()  # initialize

        # Work around https://github.com/pytorch/pytorch/issues/27972
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            f = io.BytesIO()
            torch.save(module, f)
            pyro.clear_param_store()
            f.seek(0)
            actual = torch.load(f, weights_only=False)

        assert_equal(actual.x, module.x)
        assert_equal(actual.y, module.y)
        assert_equal(actual.z, module.z)
        assert actual.s.shape == module.s.shape
        assert actual.t.shape == module.t.shape
        actual_names = {name for name, _ in actual.named_parameters()}
        expected_names = {name for name, _ in module.named_parameters()}
        assert actual_names == expected_names

        actual()
        if local_params:
            assert len(set(pyro.get_param_store().keys())) == 0
            assert torch.all(module.y != module2.y)
            assert torch.all(actual.y != module2.y)
        else:
            assert len(set(pyro.get_param_store().keys())) > 0
            assert_equal(module.y, module2.y)
            assert_equal(actual.y, module2.y)


def test_pyro_serialize():
    class MyModule(PyroModule):
        def __init__(self):
            super().__init__()
            self.x = PyroParam(torch.tensor(1.234), constraints.positive)
            self.y = nn.Parameter(torch.randn(3))

        def forward(self):
            return self.x, self.y

    module = MyModule()
    assert len(pyro.get_param_store()) == 0

    assert isinstance(module.x, torch.Tensor)
    assert len(pyro.get_param_store()) == 0

    actual = module()  # triggers saving in param store
    assert_equal(actual[0], module.x)
    assert_equal(actual[1], module.y)
    assert set(pyro.get_param_store().keys()) == {"x", "y"}
    assert_equal(pyro.param("x").detach(), module.x.detach())
    assert_equal(pyro.param("y").detach(), module.y.detach())

    pyro.get_param_store().save("/tmp/pyro_module.pt")
    pyro.clear_param_store()
    assert len(pyro.get_param_store()) == 0
    pyro.get_param_store().load("/tmp/pyro_module.pt")
    assert set(pyro.get_param_store().keys()) == {"x", "y"}
    actual = MyModule()
    actual()

    assert_equal(actual.x, module.x)
    actual_names = {name for name, _ in actual.named_parameters()}
    expected_names = {name for name, _ in module.named_parameters()}
    assert actual_names == expected_names


def test_bayesian_gru():
    input_size = 2
    hidden_size = 3
    batch_size = 4
    seq_len = 5

    # Construct a simple GRU.
    gru = nn.GRU(input_size, hidden_size)
    input_ = torch.randn(seq_len, batch_size, input_size)
    output, _ = gru(input_)
    assert output.shape == (seq_len, batch_size, hidden_size)
    output2, _ = gru(input_)
    assert torch.allclose(output2, output)

    # Make it Bayesian.
    to_pyro_module_(gru)
    for name, value in list(gru.named_parameters(recurse=False)):
        prior = dist.Normal(0, 1).expand(value.shape).to_event(value.dim())
        setattr(gru, name, PyroSample(prior=prior))
    output, _ = gru(input_)
    assert output.shape == (seq_len, batch_size, hidden_size)
    output2, _ = gru(input_)
    assert not torch.allclose(output2, output)


@pytest.mark.parametrize(
    "use_local_params",
    [
        True,
        xfail_param(
            False, reason="torch.func not compatible with global parameter store"
        ),
    ],
)
def test_functorch_pyroparam(use_local_params):
    class ParamModule(PyroModule):
        def __init__(self):
            super().__init__()
            self.a2 = PyroParam(torch.tensor(0.678), constraints.positive)

        @PyroParam(constraint=constraints.real)
        def a1(self):
            return torch.tensor(0.456)

    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.param_module = ParamModule()
            self.b1 = PyroParam(torch.tensor(0.123), constraints.positive)
            self.b3 = torch.nn.Parameter(torch.tensor(0.789))
            self.c = torch.nn.Linear(1, 1)

        @PyroParam(constraint=constraints.positive)
        def b2(self):
            return torch.tensor(1.234)

        def forward(self, x, y):
            return (
                (self.param_module.a1 + self.param_module.a2) * x
                + self.b1
                + self.b2
                + self.b3
                - self.c(y.unsqueeze(-1)).squeeze(-1)
            ) ** 2

    with pyro.settings.context(module_local_params=use_local_params):
        model = Model()
        x, y = torch.tensor(1.3), torch.tensor(0.2)

        with pyro.poutine.trace() as tr:
            model(x, y)

        params = dict(model.named_parameters())

        # Check that all parameters appear in the trace for SVI compatibility
        assert len(params) == len(
            {
                name: node
                for name, node in tr.trace.nodes.items()
                if node["type"] == "param"
            }
        )

        grad_model = torch.func.grad(
            lambda p, x, y: torch.func.functional_call(model, p, (x, y))
        )
        grad_params_func = grad_model(params, x, y)

        gs = torch.autograd.grad(model(x, y), tuple(params.values()))
        grad_params_autograd = dict(zip(params.keys(), gs))

        assert len(grad_params_autograd) == len(grad_params_func) != 0
        assert (
            set(grad_params_autograd.keys())
            == set(grad_params_func.keys())
            == set(params.keys())
        )
        for k in grad_params_autograd.keys():
            assert not torch.allclose(
                grad_params_func[k], torch.zeros_like(grad_params_func[k])
            ), k
            assert torch.allclose(grad_params_autograd[k], grad_params_func[k]), k


class BNN(PyroModule):
    # this is a vanilla Bayesian neural network implementation, nothing new or exiting here
    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: Iterable[int],
        output_size: int,
        use_new_module_list_type: bool,
    ) -> None:
        super().__init__()

        layer_sizes = (
            [(input_size, hidden_layer_sizes[0])]
            + list(zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:]))
            + [(hidden_layer_sizes[-1], output_size)]
        )

        layers = [
            pyro.nn.module.PyroModule[torch.nn.Linear](in_size, out_size)
            for in_size, out_size in layer_sizes
        ]
        if use_new_module_list_type:
            self.layers = pyro.nn.module.PyroModuleList(layers)
        else:
            self.layers = pyro.nn.module.PyroModule[torch.nn.ModuleList](layers)

        # make the layers Bayesian
        for layer_idx, layer in enumerate(self.layers):
            layer.weight = pyro.nn.module.PyroSample(
                dist.Normal(0.0, 5.0 * math.sqrt(2 / layer_sizes[layer_idx][0]))
                .expand(
                    [
                        layer_sizes[layer_idx][1],
                        layer_sizes[layer_idx][0],
                    ]
                )
                .to_event(2)
            )
            layer.bias = pyro.nn.module.PyroSample(
                dist.Normal(0.0, 5.0).expand([layer_sizes[layer_idx][1]]).to_event(1)
            )

        self.activation = torch.nn.Tanh()
        self.output_size = output_size

    def forward(self, x: torch.Tensor, obs=None) -> torch.Tensor:
        mean = self.layers[-1](x)

        if obs is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample(
                    "obs", dist.Normal(mean, 0.1).to_event(self.output_size), obs=obs
                )

        return mean


class SliceIndexingModuleListBNN(BNN):
    # I claim that it makes a difference whether slice-indexing is used or whether position-indexing is used
    # when sub-pyromodule are wrapped in a PyroModule[torch.nn.ModuleList]
    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: Iterable[int],
        output_size: int,
        use_new_module_list_type: bool,
    ) -> None:
        super().__init__(
            input_size, hidden_layer_sizes, output_size, use_new_module_list_type
        )

    def forward(self, x: torch.Tensor, obs=None) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        return super().forward(x, obs=obs)


class PositionIndexingModuleListBNN(BNN):
    # I claim that it makes a difference whether slice-indexing is used or whether position-indexing is used
    # when sub-pyromodule are wrapped in a PyroModule[torch.nn.ModuleList]
    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: Iterable[int],
        output_size: int,
        use_new_module_list_type: bool,
    ) -> None:
        super().__init__(
            input_size, hidden_layer_sizes, output_size, use_new_module_list_type
        )

    def forward(self, x: torch.Tensor, obs=None) -> torch.Tensor:
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        return super().forward(x, obs=obs)


class NestedBNN(pyro.nn.module.PyroModule):
    # finally, the issue I want to describe occurs after the second "layer of nesting",
    # i.e. when a PyroModule[ModuleList] is wrapped in a PyroModule[ModuleList]
    def __init__(self, bnns: Iterable[BNN], use_new_module_list_type: bool) -> None:
        super().__init__()
        if use_new_module_list_type:
            self.bnns = pyro.nn.module.PyroModuleList(bnns)
        else:
            self.bnns = pyro.nn.module.PyroModule[torch.nn.ModuleList](bnns)

    def forward(self, x: torch.Tensor, obs=None) -> torch.Tensor:
        mean = sum([bnn(x) for bnn in self.bnns]) / len(self.bnns)

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mean, 0.1).to_event(1), obs=obs)

        return mean


def train_bnn(model: BNN, input_size: int) -> None:
    pyro.clear_param_store()

    # small numbers for demo purposes
    num_points = 20
    num_svi_iterations = 100

    x = torch.linspace(0, 1, num_points).reshape((-1, input_size))
    y = torch.sin(2 * math.pi * x) + torch.randn(x.size()) * 0.1

    guide = AutoDiagonalNormal(model)
    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    for _ in range(num_svi_iterations):
        svi.step(x, y)


class ModuleListTester:
    def setup(self, use_new_module_list_type: bool) -> None:
        self.input_size = 1
        self.output_size = 1
        self.hidden_size = 3
        self.num_hidden_layers = 3
        self.use_new_module_list_type = use_new_module_list_type

    def get_position_indexing_modulelist_bnn(self) -> PositionIndexingModuleListBNN:
        return PositionIndexingModuleListBNN(
            self.input_size,
            [self.hidden_size] * self.num_hidden_layers,
            self.output_size,
            self.use_new_module_list_type,
        )

    def get_slice_indexing_modulelist_bnn(self) -> SliceIndexingModuleListBNN:
        return SliceIndexingModuleListBNN(
            self.input_size,
            [self.hidden_size] * self.num_hidden_layers,
            self.output_size,
            self.use_new_module_list_type,
        )

    def train_nested_bnn(self, module_getter: Callable[[], BNN]) -> None:
        train_bnn(
            NestedBNN(
                [module_getter() for _ in range(2)],
                use_new_module_list_type=self.use_new_module_list_type,
            ),
            self.input_size,
        )


class TestTorchModuleList(ModuleListTester):
    def test_with_position_indexing(self) -> None:
        self.setup(False)
        self.train_nested_bnn(self.get_position_indexing_modulelist_bnn)

    def test_with_slice_indexing(self) -> None:
        self.setup(False)
        # with pytest.raises(RuntimeError):
        # error no longer gets raised
        self.train_nested_bnn(self.get_slice_indexing_modulelist_bnn)


class TestPyroModuleList(ModuleListTester):
    def test_with_position_indexing(self) -> None:
        self.setup(True)
        self.train_nested_bnn(self.get_position_indexing_modulelist_bnn)

    def test_with_slice_indexing(self) -> None:
        self.setup(True)
        self.train_nested_bnn(self.get_slice_indexing_modulelist_bnn)


def test_module_list() -> None:
    assert PyroModule[torch.nn.ModuleList] is pyro.nn.PyroModuleList


@pytest.mark.parametrize("use_module_local_params", [True, False])
def test_render_constrained_param(use_module_local_params):

    class Model(PyroModule):

        @PyroParam(constraint=constraints.positive)
        def x(self):
            return torch.tensor(1.234)

        @PyroParam(constraint=constraints.real)
        def y(self):
            return torch.tensor(0.456)

        def forward(self):
            return self.x + self.y

    with pyro.settings.context(module_local_params=use_module_local_params):
        model = Model()
        pyro.render_model(model)
