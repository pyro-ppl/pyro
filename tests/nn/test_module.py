# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import io
import warnings

import pytest
import torch
from torch import nn
from torch.distributions import constraints, transform_to

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.nn.module import PyroModule, PyroParam, PyroSample, clear, to_pyro_module_
from pyro.optim import Adam
from tests.common import assert_equal


def test_svi_smoke():

    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.loc = nn.Parameter(torch.zeros(2))
            self.scale = PyroParam(torch.ones(2), constraint=constraints.positive)
            self.z = PyroSample(lambda self: dist.Normal(self.loc, self.scale).to_event(1))

        def forward(self, data):
            loc, log_scale = self.z.unbind(-1)
            with pyro.plate("data"):
                pyro.sample("obs", dist.Cauchy(loc, log_scale.exp()),
                            obs=data)

    class Guide(PyroModule):
        def __init__(self):
            super().__init__()
            self.loc = nn.Parameter(torch.zeros(2))
            self.scale = PyroParam(torch.ones(2), constraint=constraints.positive)
            self.z = PyroSample(lambda self: dist.Normal(self.loc, self.scale).to_event(1))

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


def test_names():

    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor(0.))
            self.y = PyroParam(torch.tensor(1.), constraint=constraints.positive)
            self.m = nn.Module()
            self.m.u = nn.Parameter(torch.tensor(2.0))
            self.p = PyroModule()
            self.p.v = nn.Parameter(torch.tensor(3.))
            self.p.w = PyroParam(torch.tensor(4.), constraint=constraints.positive)

        def forward(self):
            # trigger .__getattr__()
            self.x
            self.y
            self.m
            self.p.v
            self.p.w

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
    actual = {name for name, site in param_capture.trace.nodes.items()
              if site["type"] == "param"}
    assert actual == expected

    # Check pyro_parameters method
    expected = {"x", "y", "m.u", "p.v", "p.w"}
    actual = set(k for k, v in model.named_pyro_params())
    assert actual == expected


def test_delete():
    m = PyroModule()
    m.a = PyroParam(torch.tensor(1.))
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
            self.child1 = Child(torch.tensor(1.))
            self.child2 = Child(torch.tensor(2.))

    f = Family()
    assert_equal(f.child1.a.detach(), torch.tensor(1.))
    assert_equal(f.child2.a.detach(), torch.tensor(2.))


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
    assert_equal(f().detach(), torch.tensor(1.))
    f.c = Child(3.)
    assert_equal(f().detach(), torch.tensor(3.))
    assert_equal(f.c().detach(), torch.tensor(3.))


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
    ((2, 5,), constraints.simplex),
    ((5, 5), constraints.lower_cholesky),
    ((2, 5, 5), constraints.lower_cholesky),
    ((10, ), constraints.greater_than(-torch.randn(10).exp())),
    ((4, 10), constraints.greater_than(-torch.randn(10).exp())),
    ((4, 10), constraints.greater_than(-torch.randn(4, 10).exp())),
    ((3, 2, 10), constraints.greater_than(-torch.randn(10).exp())),
    ((3, 2, 10), constraints.greater_than(-torch.randn(2, 10).exp())),
    ((3, 2, 10), constraints.greater_than(-torch.randn(3, 1, 10).exp())),
    ((3, 2, 10), constraints.greater_than(-torch.randn(3, 2, 10).exp())),
    ((5,), constraints.real_vector),
    ((2, 5,), constraints.real_vector),
    ((), constraints.unit_interval),
    ((4, ), constraints.unit_interval),
    ((3, 2), constraints.unit_interval),
    ((10,), constraints.interval(-torch.randn(10).exp(),
                                 torch.randn(10).exp())),
    ((4, 10), constraints.interval(-torch.randn(10).exp(),
                                   torch.randn(10).exp())),
    ((3, 2, 10), constraints.interval(-torch.randn(10).exp(),
                                      torch.randn(10).exp())),
]


@pytest.mark.parametrize('shape,constraint_', SHAPE_CONSTRAINT)
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
    assert 'x' not in module._pyro_params
    assert not hasattr(module, 'x')
    assert not hasattr(module, 'x_unconstrained')


def test_clear():

    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor(0.))
            self.m = torch.nn.Linear(2, 3)
            self.m.weight.data.fill_(1.)
            self.m.bias.data.fill_(2.)
            self.p = PyroModule()
            self.p.x = nn.Parameter(torch.tensor(3.))

        def forward(self):
            return [x.clone() for x in [self.x, self.m.weight, self.m.bias, self.p.x]]

    assert set(pyro.get_param_store().keys()) == set()
    m = Model()
    state0 = m()
    assert set(pyro.get_param_store().keys()) == {"x", "m$$$weight", "m$$$bias", "p.x"}

    # mutate
    for x in pyro.get_param_store().values():
        x.unconstrained().data += torch.randn(())
    state1 = m()
    for x, y in zip(state0, state1):
        assert not (x == y).all()
    assert set(pyro.get_param_store().keys()) == {"x", "m$$$weight", "m$$$bias", "p.x"}

    clear(m)
    del m
    assert set(pyro.get_param_store().keys()) == set()

    m = Model()
    state2 = m()
    assert set(pyro.get_param_store().keys()) == {"x", "m$$$weight", "m$$$bias", "p.x"}
    for actual, expected in zip(state2, state0):
        assert_equal(actual, expected)


def test_sample():

    class Model(nn.Linear, PyroModule):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features)
            self.weight = PyroSample(
                lambda self: dist.Normal(0, 1)
                                 .expand([self.out_features,
                                          self.in_features])
                                 .to_event(2))

    class Guide(nn.Linear, PyroModule):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features)
            self.loc = PyroParam(torch.zeros_like(self.weight))
            self.scale = PyroParam(torch.ones_like(self.weight),
                                   constraint=constraints.positive)
            self.weight = PyroSample(
                lambda self: dist.Normal(self.loc, self.scale)
                                 .to_event(2))

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
    module.a = nn.Parameter(torch.tensor(0.))
    module.b = PyroParam(torch.tensor(1.), constraint=constraints.positive)
    module.c = PyroSample(dist.Normal(0, 1))
    module.p = PyroModule()
    module.p.d = nn.Parameter(torch.tensor(3.))
    module.p.e = PyroParam(torch.tensor(4.), constraint=constraints.positive)
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
        self.z = PyroParam(torch.ones(size),
                           constraint=constraints.positive,
                           event_dim=1)
        self.s = PyroSample(dist.Normal(0, 1))
        self.t = PyroSample(lambda self: dist.Normal(self.s, self.z))

    def forward(self):
        return self.x + self.y + self.t


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

    def forward(self):
        return self.x + self.y + self.t


@pytest.mark.parametrize("Model", [AttributeModel, DecoratorModel])
@pytest.mark.parametrize("size", [1, 2])
def test_decorator(Model, size):
    model = Model(size)
    for i in range(2):
        trace = poutine.trace(model).get_trace()
        assert set(trace.nodes.keys()) == {"_INPUT", "x", "y", "z", "s", "t", "_RETURN"}

        assert trace.nodes["x"]["type"] == "param"
        assert trace.nodes["y"]["type"] == "param"
        assert trace.nodes["z"]["type"] == "param"
        assert trace.nodes["s"]["type"] == "sample"
        assert trace.nodes["t"]["type"] == "sample"

        assert trace.nodes["x"]["value"].shape == (size,)
        assert trace.nodes["y"]["value"].shape == (size,)
        assert trace.nodes["z"]["value"].shape == (size,)
        assert trace.nodes["s"]["value"].shape == ()
        assert trace.nodes["t"]["value"].shape == (size,)


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
        module = torch.load(f)
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
                setattr(m, name, PyroSample(prior=dist.Normal(0, 1)
                                                      .expand(value.shape)
                                                      .to_event(value.dim())))
    randomize(actual)
    randomize(expected)
    assert_identical(actual, expected)


def test_torch_serialize_attributes():
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
        actual = torch.load(f)

    assert_equal(actual.x, module.x)
    actual_names = {name for name, _ in actual.named_parameters()}
    expected_names = {name for name, _ in module.named_parameters()}
    assert actual_names == expected_names


def test_torch_serialize_decorators():
    module = DecoratorModel(3)
    module()  # initialize

    # Work around https://github.com/pytorch/pytorch/issues/27972
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        f = io.BytesIO()
        torch.save(module, f)
        pyro.clear_param_store()
        f.seek(0)
        actual = torch.load(f)

    assert_equal(actual.x, module.x)
    assert_equal(actual.y, module.y)
    assert_equal(actual.z, module.z)
    assert actual.s.shape == module.s.shape
    assert actual.t.shape == module.t.shape
    actual_names = {name for name, _ in actual.named_parameters()}
    expected_names = {name for name, _ in module.named_parameters()}
    assert actual_names == expected_names


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
