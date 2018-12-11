from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoname import named


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
        loc = latent.loc.param_(torch.zeros(1))
        foo = latent.foo.sample_(dist.Normal(loc, torch.ones(1)))
        latent.bar.sample_(dist.Normal(loc, torch.ones(1)), obs=foo)
        latent.x.z.sample_(dist.Normal(loc, torch.ones(1)))

    tr = poutine.trace(model).get_trace()
    assert get_sample_names(tr) == set(["latent.foo", "latent.x.z"])
    assert get_observe_names(tr) == set(["latent.bar"])
    assert get_param_names(tr) == set(["latent.loc"])


def test_named_list():
    pyro.clear_param_store()

    def model():
        latent = named.List("latent")
        loc = latent.add().param_(torch.zeros(1))
        foo = latent.add().sample_(dist.Normal(loc, torch.ones(1)))
        latent.add().sample_(dist.Normal(loc, torch.ones(1)), obs=foo)
        latent.add().z.sample_(dist.Normal(loc, torch.ones(1)))

    tr = poutine.trace(model).get_trace()
    assert get_sample_names(tr) == set(["latent[1]", "latent[3].z"])
    assert get_observe_names(tr) == set(["latent[2]"])
    assert get_param_names(tr) == set(["latent[0]"])


def test_named_dict():
    pyro.clear_param_store()

    def model():
        latent = named.Dict("latent")
        loc = latent["loc"].param_(torch.zeros(1))
        foo = latent["foo"].sample_(dist.Normal(loc, torch.ones(1)))
        latent["bar"].sample_(dist.Normal(loc, torch.ones(1)), obs=foo)
        latent["x"].z.sample_(dist.Normal(loc, torch.ones(1)))

    tr = poutine.trace(model).get_trace()
    assert get_sample_names(tr) == set(["latent['foo']", "latent['x'].z"])
    assert get_observe_names(tr) == set(["latent['bar']"])
    assert get_param_names(tr) == set(["latent['loc']"])


def test_nested():
    pyro.clear_param_store()

    def model():
        latent = named.Object("latent")
        latent.list = named.List()
        loc = latent.list.add().loc.param_(torch.zeros(1))
        latent.dict = named.Dict()
        foo = latent.dict["foo"].foo.sample_(dist.Normal(loc, torch.ones(1)))
        latent.object.bar.sample_(dist.Normal(loc, torch.ones(1)), obs=foo)

    tr = poutine.trace(model).get_trace()
    assert get_sample_names(tr) == set(["latent.dict['foo'].foo"])
    assert get_observe_names(tr) == set(["latent.object.bar"])
    assert get_param_names(tr) == set(["latent.list[0].loc"])


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
