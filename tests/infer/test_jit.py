from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from tests.common import assert_equal


def test_simple():
    y = torch.ones(2)

    @torch.jit.compile(nderivs=0)
    def f(x):
        print('Inside f')
        assert x is y
        return y + 1.0

    print('Calling f(y)')
    assert_equal(f(y), y.new_tensor([2, 2]))
    print('Calling f(y)')
    assert_equal(f(y), y.new_tensor([2, 2]))
    print('Calling f(torch.zeros(2))')
    assert_equal(f(torch.zeros(2)), y.new_tensor([1, 1]))
    with pytest.raises(AssertionError):
        assert_equal(f(torch.ones(5)), y.new_tensor([2, 2, 2, 2, 2]))


def test_backward():
    y = torch.ones(2, requires_grad=True)

    @torch.jit.compile(nderivs=1)
    def f(x):
        print('Inside f')
        assert x is y
        return (y + 1.0).sum()

    print('Calling f(y)')
    f(y).backward()
    print('Calling f(y)')
    f(y)
    print('Calling f(torch.zeros(2))')
    f(torch.zeros(2, requires_grad=True))
    with pytest.raises(AssertionError):
        f(torch.ones(5, requires_grad=True))


@pytest.mark.parametrize('use_jit', [False, True])
def test_traceenum(use_jit):
    pyro.clear_param_store()
    data = torch.arange(10)

    def model(data):
        loc = pyro.param("loc", torch.tensor(0.0))
        scale = pyro.param("scale", torch.tensor(1.0), constraint=constraints.positive)
        pyro.sample("x", dist.Normal(loc, scale).expand_by(data.shape).independent(1), obs=data)

    def guide(data):
        pass

    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    loss_and_grads = elbo.jit_loss_and_grads if use_jit else elbo.loss_and_grads
    inference = SVI(model, guide, Adam({"lr": 1e-6}),
                    loss=elbo.loss,
                    loss_and_grads=loss_and_grads)
    for i in range(1000):
        inference.step(data)
