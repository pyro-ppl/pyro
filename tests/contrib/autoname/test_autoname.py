# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions.torch as dist
import pyro.poutine as poutine
from pyro.contrib.autoname import autoname, sample


def test_basic_scope():
    @autoname
    def f1():
        sample(dist.Normal(0, 1))
        return sample(dist.Bernoulli(0.5))

    @autoname(name="model")
    def f2():
        sample("x", dist.Bernoulli(0.5))
        return sample(dist.Normal(0.0, 1.0))

    tr1 = poutine.trace(f1).get_trace()
    assert "f1/Normal" in tr1.nodes
    assert "f1/Bernoulli" in tr1.nodes

    tr2 = poutine.trace(f2).get_trace()
    assert "model/x" in tr2.nodes
    assert "model/Normal" in tr2.nodes


def test_repeat_names():
    @autoname
    def f1():
        sample(dist.Normal(0, 1))
        sample(dist.Normal(0, 1))
        return sample(dist.Bernoulli(0.5))

    @autoname(name="model")
    def f2():
        sample("x", dist.Bernoulli(0.5))
        sample("x", dist.Bernoulli(0.5))
        sample("x", dist.Bernoulli(0.5))
        return sample(dist.Normal(0.0, 1.0))

    tr1 = poutine.trace(f1).get_trace()
    assert "f1/Normal" in tr1.nodes
    assert "f1/Normal1" in tr1.nodes
    assert "f1/Bernoulli" in tr1.nodes

    tr2 = poutine.trace(f2).get_trace()
    assert "model/x" in tr2.nodes
    assert "model/x1" in tr2.nodes
    assert "model/x2" in tr2.nodes
    assert "model/Normal" in tr2.nodes


def test_compose_scopes():
    @autoname
    def f1():
        return sample(dist.Bernoulli(0.5))

    @autoname
    def f2():
        f1()
        return sample(dist.Bernoulli(0.5))

    @autoname
    def f3():
        f1()
        f1()
        f1()
        f2()
        return sample(dist.Normal(0, 1))

    tr1 = poutine.trace(f1).get_trace()
    assert "f1/Bernoulli" in tr1.nodes

    tr2 = poutine.trace(f2).get_trace()
    assert "f2/f1/Bernoulli" in tr2.nodes
    assert "f2/Bernoulli" in tr2.nodes

    tr3 = poutine.trace(f3).get_trace()
    assert "f3/f1/Bernoulli" in tr3.nodes
    assert "f3/f1__1/Bernoulli" in tr3.nodes
    assert "f3/f1__2/Bernoulli" in tr3.nodes
    assert "f3/f2/f1/Bernoulli" in tr3.nodes
    assert "f3/f2/Bernoulli" in tr3.nodes
    assert "f3/Normal" in tr3.nodes


def test_basic_loop():
    @autoname
    def f1():
        return sample(dist.Bernoulli(0.5))

    @autoname(name="model")
    def f2():
        f1()
        for i in range(3):
            f1()
            sample("x", dist.Bernoulli(0.5))
        return sample(dist.Normal(0.0, 1.0))

    tr = poutine.trace(f2).get_trace()
    assert "model/f1/Bernoulli" in tr.nodes
    assert "model/f1__1/Bernoulli" in tr.nodes
    assert "model/f1__2/Bernoulli" in tr.nodes
    assert "model/f1__3/Bernoulli" in tr.nodes
    assert "model/x" in tr.nodes
    assert "model/x1" in tr.nodes
    assert "model/x2" in tr.nodes
    assert "model/Normal" in tr.nodes


def test_named_loop():
    @autoname
    def f1():
        return sample(dist.Bernoulli(0.5))

    @autoname(name="model")
    def f2():
        f1()
        for i in autoname(name="loop")(range(3)):
            f1()
            sample("x", dist.Bernoulli(0.5))
        return sample(dist.Normal(0.0, 1.0))

    tr = poutine.trace(f2).get_trace()
    assert "model/f1/Bernoulli" in tr.nodes
    assert "model/loop/f1/Bernoulli" in tr.nodes
    assert "model/loop__1/f1/Bernoulli" in tr.nodes
    assert "model/loop__2/f1/Bernoulli" in tr.nodes
    assert "model/loop/x" in tr.nodes
    assert "model/loop__1/x" in tr.nodes
    assert "model/loop__2/x" in tr.nodes
    assert "model/Normal" in tr.nodes


def test_sequential_plate():
    @autoname
    def f1():
        return sample(dist.Bernoulli(0.5))

    @autoname(name="model")
    def f2():
        for i in autoname(pyro.plate(name="data", size=3)):
            f1()
        return sample(dist.Bernoulli(0.5))

    expected_names = [
        "model/data/f1/Bernoulli",
        "model/data__1/f1/Bernoulli",
        "model/data__2/f1/Bernoulli",
        "model/Bernoulli",
    ]

    tr = poutine.trace(f2).get_trace()
    actual_names = [
        name
        for name, node in tr.nodes.items()
        if node["type"] == "sample" and type(node["fn"]).__name__ != "_Subsample"
    ]
    assert expected_names == actual_names
