import warnings

import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints, transform_to

from pyro.params import ConstrainedParameter, constraint
from tests.common import assert_equal

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
def test_constrained_parameter(shape, constraint_):

    class MyModule(torch.nn.Module):
        @constraint
        def x(self):
            return constraint_

    module = MyModule()
    module.x = torch.full(shape, 1e-4)

    assert isinstance(MyModule.x, ConstrainedParameter)
    assert isinstance(module.x, torch.Tensor)
    assert module.x.shape == shape
    assert constraint_.check(module.x).all()

    module.x = torch.randn(shape).exp() * 1e-6
    assert isinstance(MyModule.x, ConstrainedParameter)
    assert isinstance(module.x, torch.Tensor)
    assert module.x.shape == shape
    assert constraint_.check(module.x).all()

    assert isinstance(module.x_unconstrained, torch.Tensor)
    y = module.x_unconstrained.data.normal_()
    assert_equal(module.x.data, transform_to(constraint_)(y))
    assert constraint_.check(module.x).all()


def test_chaining():

    class ChainedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.randn(()).exp().requires_grad_()
            self.y = self.x + torch.randn(()).exp()
            self.z = self.y + torch.randn(()).exp()

        @constraint
        def y(self):
            return constraints.greater_than(self.x)

        @constraint
        def z(self):
            return constraints.greater_than(self.y)

    module = ChainedModule()

    dy_dx, = grad(module.y, [module.x])
    assert (dy_dx > 0).all()

    dz_dy, = grad(module.z, [module.y_unconstrained])
    assert (dz_dy > 0).all()

    dz_dx, = grad(module.z, [module.x])
    assert (dz_dx > 0).all()


class PositiveModule(torch.nn.Module):
    @constraint
    def x(self):
        return constraints.positive


def test_serialization():

    module = PositiveModule()
    module.x = torch.tensor(1.234)
    assert isinstance(module.x, torch.Tensor)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.save(module, "/tmp/test_constrained_parameter.pkl")
        actual = torch.load("/tmp/test_constrained_parameter.pkl")
    assert_equal(actual.x, module.x)
