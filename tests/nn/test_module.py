import warnings

import pytest
import torch
from torch import nn
from torch.distributions import constraints, transform_to

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.nn.module import PyroModule, PyroParam, PyroSample
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
    assert "scale_unconstrained" in trace.nodes
    assert trace.nodes["scale_unconstrained"]["type"] == "param"

    guide = Guide()
    trace = poutine.trace(guide).get_trace(data)
    assert "loc" in trace.nodes.keys()
    assert trace.nodes["loc"]["type"] == "param"
    assert "scale_unconstrained" in trace.nodes
    assert trace.nodes["scale_unconstrained"]["type"] == "param"

    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, Trace_ELBO())
    for step in range(3):
        svi.step(data)


def test_names():
    root = PyroModule()
    root.x = nn.Parameter(torch.tensor(0.))
    root.y = PyroParam(torch.tensor(1.), constraint=constraints.positive)
    root.m = nn.Module()
    root.m.u = nn.Parameter(torch.tensor(2.0))
    root.p = PyroModule()
    root.p.v = nn.Parameter(torch.tensor(3.))
    root.p.w = PyroParam(torch.tensor(4.), constraint=constraints.positive)

    # Check named_parameters.
    expected = {
        "x",
        "y_unconstrained",
        "m.u",
        "p.v",
        "p.w_unconstrained",
    }
    actual = set(name for name, _ in root.named_parameters())
    assert actual == expected

    # Check pyro.param names.
    expected = {
        "x",
        "y_unconstrained",
        "m$$$u",
        "p.v",
        "p.w_unconstrained",
    }
    with poutine.trace(param_only=True) as param_capture:
        # trigger .__getattr__()
        root.x
        root.y
        root.m
        root.p.v
        root.p.w
    actual = {name for name, site in param_capture.trace.nodes.items()
              if site["type"] == "param"}
    assert actual == expected


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

    assert module._pyro_cache is module.p._pyro_cache

    # Check that results are cached with an invocation of .__call__().
    result1 = module()
    actual, expected = result1
    for key in ["a", "c", "p.d", "p.f"]:
        assert actual[key] is expected[key], key

    # Check that results are not cached across invocations of .__call__().
    result2 = module()
    for key in ["b", "c", "p.e", "p.f"]:
        assert result1[0] is not result2[0], key


def test_serialization():

    module = PyroModule()
    module.x = PyroParam(torch.tensor(1.234), constraints.positive)
    assert isinstance(module.x, torch.Tensor)

    # Work around https://github.com/pytorch/pytorch/issues/27972
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.save(module, "/tmp/test_pyro_module.pkl")
        actual = torch.load("/tmp/test_pyro_module.pkl")
    assert_equal(actual.x, module.x)
    actual_names = {name for name, _ in actual.named_parameters()}
    expected_names = {name for name, _ in module.named_parameters()}
    assert actual_names == expected_names
