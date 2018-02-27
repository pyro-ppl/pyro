from __future__ import absolute_import, division, print_function

import gc

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.svi import SVI
from pyro.optim import Adam
from pyro.poutine.trace import Trace


pytestmark = pytest.mark.stage('unit')


def count_objects_of_type(type_):
    return sum(1 for obj in gc.get_objects() if isinstance(obj, type_))


def test_trace():
    n = 11
    data = Variable(torch.zeros(n))

    def model(data):
        loc = pyro.param('loc', Variable(torch.zeros(n), requires_grad=True))
        scale = pyro.param('log_scale', Variable(torch.zeros(n), requires_grad=True)).exp()
        pyro.sample('obs', dist.Normal(loc, scale), obs=data)

    counts = []
    gc.collect()
    gc.collect()
    expected = count_objects_of_type(Trace)
    for _ in range(10):
        poutine.trace(model)(data)
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([expected]), counts


class Foo(dict):
    pass


def test_dict_copy():
    counts = []
    gc.collect()
    gc.collect()
    f = Foo()
    for _ in range(10):
        f.copy()
        counts.append(count_objects_of_type(Foo))

    assert set(counts) == set([1]), counts


def test_copy():
    counts = []
    gc.collect()
    gc.collect()
    tr = Trace()
    expected = count_objects_of_type(Trace)
    for _ in range(10):
        tr.copy()
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([expected]), counts


def test_trace_copy():
    n = 11
    data = Variable(torch.zeros(n))

    def model(data):
        loc = pyro.param('loc', Variable(torch.zeros(n), requires_grad=True))
        scale = pyro.param('log_scale', Variable(torch.zeros(n), requires_grad=True)).exp()
        pyro.sample('obs', dist.Normal(loc, scale), obs=data)

    counts = []
    gc.collect()
    gc.collect()
    expected = count_objects_of_type(Trace)
    for _ in range(10):
        poutine.trace(model).get_trace(data).copy()
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([expected]), counts


def trace_replay(model, guide, *args):
    guide_trace = poutine.trace(guide).get_trace(*args)
    poutine.trace(poutine.replay(model, guide_trace)).get_trace(*args)


def test_trace_replay():
    n = 11
    data = Variable(torch.zeros(n))

    def model(data):
        loc = pyro.param('loc', Variable(torch.zeros(n), requires_grad=True))
        scale = pyro.param('log_scale', Variable(torch.zeros(n), requires_grad=True)).exp()
        pyro.sample('obs', dist.Normal(loc, scale), obs=data)

    def guide(data):
        pass

    counts = []
    gc.collect()
    gc.collect()
    expected = count_objects_of_type(Trace)
    for _ in range(10):
        trace_replay(model, guide, data)
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([expected]), counts


def test_svi():
    n = 11
    data = Variable(torch.zeros(n))

    def model(data):
        loc = pyro.param('loc', Variable(torch.zeros(n), requires_grad=True))
        scale = pyro.param('log_scale', Variable(torch.zeros(n), requires_grad=True)).exp()
        pyro.sample('obs', dist.Normal(loc, scale), obs=data)

    def guide(data):
        pass

    optim = Adam({'lr': 1e-3})
    inference = SVI(model, guide, optim, 'ELBO')

    counts = []
    gc.collect()
    gc.collect()
    expected = count_objects_of_type(Trace)
    for _ in range(10):
        inference.step(data)
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([expected]), counts
