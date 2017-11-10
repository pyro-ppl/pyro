from __future__ import absolute_import, division, print_function

import torch
import torch.nn
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib import named
import pyro.infer
from pyro.util import ng_ones

from tests.common import assert_equal


# def test_object():
#     obj = named.Object("name")
#     obj.a.b.param_(10)

#     assert pyro.param("name.a.b") == 10
#     assert pyro.param("name.a.b") == obj.a.b

#     obj.c = named.List()
#     obj.c.add().param_(20)

#     assert pyro.param("name.c[0]") == 20
#     assert pyro.param("name.c[0]") == obj.c[0]

#     obj.d.sample_(dist.normal, Variable(torch.Tensor([10])),
#                   Variable(torch.Tensor([0])))
#     assert obj.d.data[0] == 10

    var_set = obj.visit(lambda name, val, acc: acc.add(name),
                        set())
    assert set(["name.a.b", "name.c[0]", "name.e['hello']", "name.d"]) == var_set


#     var_set = obj.visit(lambda name, val, acc: acc.add(name),
#                         set())
#     assert set(["name.a.b", "name.c[0]", "name.e['hello']", "name.d"]) == var_set

#     var_dict = obj.visit(lambda name, val, acc: acc.setdefault(name, val),
#                          dict())
#     assert var_dict["name.a.b"] == 10
#     assert var_dict["name.c[0]"] == 20

    
def test_module():
    m = torch.nn.Sequential()
    m.add_module("first", torch.nn.Linear(10, 10))
    m.add_module("second", torch.nn.Linear(20, 20))
    obj = named.Object("modulewrap")
    obj.module_(m)
    assert_equal(obj.first.weight, m.first.weight)
    assert_equal(obj.second.bias, m.second.bias)
    assert_equal(pyro.param("modulewrap.first.weight"),  m.first.weight)



# def test_irange():
#     obj = named.Object("name")

#     obj.data = named.List()
#     for i, latent in obj.data.irange_(10):
#         latent.param_(i)

#     assert pyro.param("name.data[0]") == 0
#     assert pyro.param("name.data[9]") == 9


def test_iarange():
    obj = named.Object("range")

    with obj.data.iarange_(10, subsample_size=5) as (ind, latent):
        latent.x.param_(ind + 10)
#     assert_equal(obj.data.x, ind + 10)
#     assert pyro.param("range.data.x").size(0) == 5

    

def model():
    obj = named.Object("name")
    mu = obj.a.sample_(dist.normal,  Variable(torch.Tensor([10])),
                       Variable(torch.Tensor([10])))


#     obj.b.observe_(dist.normal, Variable(torch.Tensor([1])), mu,
#                    Variable(torch.Tensor([10])))


    
def guide():
    obj = named.Object("name")
    obj.a.sample_(dist.normal,  Variable(torch.Tensor([10])),
                  Variable(torch.Tensor([10])))


def test_infer():
    # For now just check if the names are matching.
    imp = pyro.infer.Importance(model, guide, num_samples=10)
    pyro.infer.Marginal(imp)()


def get_sample_names(tr):
    return set([name for name, site in tr.nodes.items()
                if site["type"] == "sample" and not site["is_observed"]])


def get_observe_names(tr):
    return set([name for name, site in tr.nodes.items()
                if site["type"] == "sample" and site["is_observed"]])


def get_param_names(tr):
    return set([name for name, site in tr.nodes.items() if site["type"] == "param"])


def test_named_object():
    pyro.clear_param_store()

    def model():
        latent = named.Object("latent")
        mu = latent.mu.param_(Variable(torch.zeros(1)))
        foo = latent.foo.sample_(dist.normal, mu, ng_ones(1))
        latent.bar.observe_(dist.normal, foo, mu, ng_ones(1))
        latent.x.z.sample_(dist.normal, mu, ng_ones(1))

    tr = poutine.trace(model).get_trace()
    assert get_sample_names(tr) == set(["latent.foo", "latent.x.z"])
    assert get_observe_names(tr) == set(["latent.bar"])
    assert get_param_names(tr) == set(["latent.mu"])


def test_named_list():
    pyro.clear_param_store()

    def model():
        latent = named.List("latent")
        mu = latent.add().param_(Variable(torch.zeros(1)))
        foo = latent.add().sample_(dist.normal, mu, ng_ones(1))
        latent.add().observe_(dist.normal, foo, mu, ng_ones(1))
        latent.add().z.sample_(dist.normal, mu, ng_ones(1))

    tr = poutine.trace(model).get_trace()
    assert get_sample_names(tr) == set(["latent[1]", "latent[3].z"])
    assert get_observe_names(tr) == set(["latent[2]"])
    assert get_param_names(tr) == set(["latent[0]"])


def test_named_dict():
    pyro.clear_param_store()

    def model():
        latent = named.Dict("latent")
        mu = latent["mu"].param_(Variable(torch.zeros(1)))
        foo = latent["foo"].sample_(dist.normal, mu, ng_ones(1))
        latent["bar"].observe_(dist.normal, foo, mu, ng_ones(1))
        latent["x"].z.sample_(dist.normal, mu, ng_ones(1))

    tr = poutine.trace(model).get_trace()
    assert get_sample_names(tr) == set(["latent['foo']", "latent['x'].z"])
    assert get_observe_names(tr) == set(["latent['bar']"])
    assert get_param_names(tr) == set(["latent['mu']"])


def test_nested():
    pyro.clear_param_store()

    def model():
        latent = named.Object("latent")
        latent.list = named.List()
        mu = latent.list.add().mu.param_(Variable(torch.zeros(1)))
        latent.dict = named.Dict()
        foo = latent.dict["foo"].foo.sample_(dist.normal, mu, ng_ones(1))
        latent.object.bar.observe_(dist.normal, foo, mu, ng_ones(1))

    tr = poutine.trace(model).get_trace()
    assert get_sample_names(tr) == set(["latent.dict['foo'].foo"])
    assert get_observe_names(tr) == set(["latent.object.bar"])
    assert get_param_names(tr) == set(["latent.list[0].mu"])


def test_eval_str():
    state = named.Object("state")
    state.x = 0
    state.ys = named.List()
    state.ys.add().foo = 1
    state.zs = named.Dict()
    state.zs[42].bar = 2

    assert state is eval(str(state))
    assert state.x is eval(str(state.x))
    assert state.ys is eval(str(state.ys))
    assert state.ys[0] is eval(str(state.ys[0]))
    assert state.ys[0].foo is eval(str(state.ys[0].foo))
    assert state.zs is eval(str(state.zs))
    assert state.zs[42] is eval(str(state.zs[42]))
    assert state.zs[42].bar is eval(str(state.zs[42].bar))
