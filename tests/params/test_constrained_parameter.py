import pytest
import torch
from torch.distributions import constraints

from pyro.params import ConstrainedParameter


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


@pytest.mark.parametrize('shape,constraint', SHAPE_CONSTRAINT)
def test_attr_of_object(shape, constraint):
    obj = type("Foo", (), {})
    p = ConstrainedParameter(torch.full(shape, 1e-4), constraint)
    obj.x = p

    assert isinstance(obj.__dict__['x'], ConstrainedParameter)
    assert isinstance(obj.x, torch.Tensor)
    assert obj.x.shape == shape
    assert constraint.check(obj.x).all()

    p.set(torch.randn(shape).exp() * 1e-6)
    assert isinstance(obj.__dict__['x'], ConstrainedParameter)
    assert isinstance(obj.x, torch.Tensor)
    assert obj.x.shape == shape
    assert constraint.check(obj.x).all()

    assert isinstance(obj.x.unconstrained(), torch.Tensor)
    obj.x.unconstrained().data.normal_()
    assert constraint.check(obj.x).all()


@pytest.mark.parametrize('shape,constraint', SHAPE_CONSTRAINT)
def test_attr_of_module(shape, constraint):
    module = type("Foo", (), {})
    p = ConstrainedParameter(torch.full(shape, 1e-4), constraint)
    module.x = p

    assert isinstance(module.__dict__['x'], ConstrainedParameter)
    assert isinstance(module.x, torch.Tensor)
    assert module.x.shape == shape
    assert constraint.check(module.x).all()

    p.set(torch.randn(shape).exp() * 1e-6)
    assert isinstance(module.__dict__['x'], ConstrainedParameter)
    assert isinstance(module.x, torch.Tensor)
    assert module.x.shape == shape
    assert constraint.check(module.x).all()

    assert isinstance(module.x.unconstrained(), torch.Tensor)
    module.x.unconstrained().data.normal_()
    assert constraint.check(module.x).all()


@pytest.mark.parametrize('shape,constraint', SHAPE_CONSTRAINT)
def test_free_object(shape, constraint):
    p = ConstrainedParameter(torch.full(shape, 1e-4), constraint)

    p.set(torch.randn(shape).exp() * 1e-6)
    assert isinstance(p.get(), torch.Tensor)
    assert p.get().shape == shape
    assert constraint.check(p.get()).all()

    assert isinstance(p.unconstrained(), torch.Tensor)
    p.unconstrained().data.normal_()
    assert constraint.check(p.get()).all()
