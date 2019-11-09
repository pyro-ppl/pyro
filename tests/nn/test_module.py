import warnings

import pytest
import torch
from torch.distributions import constraints, transform_to

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.nn.module import PyroModule, PyroParam, PyroSample
from pyro.optim import Adam
from tests.common import assert_equal


class Model(PyroModule):
    def __init__(self):
        super().__init__()
        self.loc = PyroParam(torch.zeros(2))
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
        self.loc = PyroParam(torch.zeros(2))
        self.scale = PyroParam(torch.ones(2), constraint=constraints.positive)
        self.z = PyroSample(lambda self: dist.Normal(self.loc, self.scale).to_event(1))

    def forward(self, *args, **kwargs):
        return self.z


def test_svi_smoke():
    data = torch.randn(5)
    model = Model()
    guide = Guide()

    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, Trace_ELBO())
    for step in range(3):
        svi.step(data)


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
def test_module_constraint(shape, constraint_):
    module = PyroModule()
    module.x = PyroParam(torch.full(shape, 1e-4), constraint_)

    assert isinstance(module.x, torch.Tensor)
    assert isinstance(module.x_unconstrained, torch.nn.Parameter)
    assert module.x.shape == shape
    assert constraint_.check(module.x).all()

    module.x = torch.randn(shape).exp() * 1e-6
    assert isinstance(module.x_unconstrained, torch.nn.Parameter)
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


def test_pyro_module_serialization():

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
