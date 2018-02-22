from __future__ import absolute_import, division, print_function

import gc

import networkx as nx
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.svi import SVI
from pyro.optim import Adam
from pyro.poutine.trace import Trace


def count_objects_of_type(type_):
    return sum(1 for obj in gc.get_objects() if isinstance(obj, type_))


def test_trace_memory():
    n = 11
    data = Variable(torch.zeros(n))

    def model(data):
        loc = pyro.param('loc', Variable(torch.zeros(n), requires_grad=True))
        scale = pyro.param('log_scale', Variable(torch.zeros(n), requires_grad=True)).exp()
        pyro.sample('obs', dist.Normal(loc, scale), obs=data)

    counts = []
    gc.collect()
    gc.collect()
    for _ in range(10):
        poutine.trace(model)(data)
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([0]), counts


def test_networkx_copy_memory():
    counts = []
    gc.collect()
    gc.collect()
    g = nx.DiGraph()
    for _ in range(10):
        g.copy()
        counts.append(count_objects_of_type(nx.DiGraph))

    assert set(counts) == set([1]), counts


def test_copy_memory():
    counts = []
    gc.collect()
    gc.collect()
    tr = Trace()
    for _ in range(10):
        tr.copy()
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([1]), counts


def test_trace_copy_memory():
    n = 11
    data = Variable(torch.zeros(n))

    def model(data):
        loc = pyro.param('loc', Variable(torch.zeros(n), requires_grad=True))
        scale = pyro.param('log_scale', Variable(torch.zeros(n), requires_grad=True)).exp()
        pyro.sample('obs', dist.Normal(loc, scale), obs=data)

    counts = []
    gc.collect()
    gc.collect()
    for _ in range(10):
        tr = poutine.trace(model).get_trace(data)
        tr.copy()
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([0]), counts


def trace_replay(model, guide, *args):
    guide_trace = poutine.trace(guide).get_trace(*args)
    poutine.trace(poutine.replay(model, guide_trace)).get_trace(*args)


def test_trace_replay_memory():
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
    for _ in range(10):
        trace_replay(model, guide, data)
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([0]), counts


def test_svi_memory():
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
    for _ in range(10):
        inference.step(data)
        counts.append(count_objects_of_type(Trace))

    assert set(counts) == set([0]), counts
