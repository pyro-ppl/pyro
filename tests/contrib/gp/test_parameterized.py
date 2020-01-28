# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.parameterized import Parameterized
from pyro.nn.module import PyroParam, PyroSample
from tests.common import assert_equal


def test_parameterized():
    class Linear(Parameterized):
        def __init__(self):
            super().__init__()
            self._pyro_name = "Linear"
            self.a = PyroParam(torch.tensor(1.), constraints.positive)
            self.b = PyroSample(dist.Normal(0, 1))
            self.c = PyroSample(dist.Normal(0, 1))
            self.d = PyroSample(dist.Normal(0, 4).expand([1]).to_event())
            self.e = PyroSample(dist.LogNormal(0, 1))
            self.f = PyroSample(dist.MultivariateNormal(torch.zeros(2), torch.eye(2)))
            self.g = PyroSample(dist.Exponential(1))

        def forward(self, x):
            return self.a * x + self.b + self.c + self.d + self.e + self.f + self.g + self.e

    linear = Linear()
    linear.autoguide("c", dist.Normal)
    linear.autoguide("d", dist.MultivariateNormal)
    linear.autoguide("e", dist.Normal)

    assert set(dict(linear.named_parameters()).keys()) == {
        "a_unconstrained", "b_map", "c_loc", "c_scale_unconstrained",
        "d_loc", "d_scale_tril_unconstrained",
        "e_loc", "e_scale_unconstrained", "f_map", "g_map_unconstrained"
    }

    def model(x):
        linear.mode = "model"
        return linear(x)

    def guide(x):
        linear.mode = "guide"
        return linear(x)

    model_trace = pyro.poutine.trace(model).get_trace(torch.tensor(5.))
    guide_trace = pyro.poutine.trace(guide).get_trace(torch.tensor(5.))
    for p in ["b", "c", "d"]:
        assert "Linear.{}".format(p) in model_trace.nodes
        assert "Linear.{}".format(p) in guide_trace.nodes

    assert isinstance(guide_trace.nodes["Linear.b"]["fn"], dist.Delta)
    c_dist = guide_trace.nodes["Linear.c"]["fn"]
    assert isinstance(getattr(c_dist, "base_dist", c_dist), dist.Normal)
    d_dist = guide_trace.nodes["Linear.d"]["fn"]
    assert isinstance(getattr(d_dist, "base_dist", d_dist), dist.MultivariateNormal)


def test_nested_parameterized():
    class Linear(Parameterized):
        def __init__(self, a):
            super().__init__()
            self.a = Parameter(a)

        def forward(self, x):
            return self.a * x

    class Quadratic(Parameterized):
        def __init__(self, linear1, linear2, a):
            super().__init__()
            self._pyro_name = "Quadratic"
            self.linear1 = linear1
            self.linear2 = linear2
            self.a = Parameter(a)

        def forward(self, x):
            return self.linear1(x) * x + self.linear2(self.a)

    linear1 = Linear(torch.tensor(1.))
    linear1.a = PyroSample(dist.Normal(0, 1))
    linear2 = Linear(torch.tensor(1.))
    linear2.a = PyroSample(dist.Normal(0, 1))
    q = Quadratic(linear1, linear2, torch.tensor(2.))
    q.a = PyroSample(dist.Cauchy(0, 1))

    def model(x):
        q.set_mode("model")
        return q(x)

    trace = pyro.poutine.trace(model).get_trace(torch.tensor(5.))
    assert "Quadratic.a" in trace.nodes
    assert "Quadratic.linear1.a" in trace.nodes
    assert "Quadratic.linear2.a" in trace.nodes


def test_inference():
    class Linear(Parameterized):
        def __init__(self, a):
            super().__init__()
            self.a = Parameter(a)

        def forward(self, x):
            return self.a * x

    target_a = torch.tensor(2.)
    x_train = torch.rand(100)
    y_train = target_a * x_train + torch.rand(100) * 0.001
    linear = Linear(torch.tensor(1.))
    linear.a = PyroSample(dist.Normal(0, 10))
    linear.autoguide("a", dist.Normal)

    def model(x, y):
        linear.set_mode("model")
        mu = linear(x)
        with pyro.plate("plate"):
            return pyro.sample("y", dist.Normal(mu, 0.1), obs=y)

    def guide(x, y):
        linear.set_mode("guide")
        linear._load_pyro_samples()

    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.5)

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model, guide, x_train, y_train)
        loss.backward()
        return loss

    for i in range(200):
        optimizer.step(closure)

    linear.mode = "guide"
    assert_equal(linear.a, target_a, prec=0.05)
