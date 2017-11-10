from __future__ import absolute_import, division, print_function

import torch
import torch.nn
from torch.autograd import Variable
import pyro.contrib.named as named
import pyro.contrib.scoped as scoped

import pyro.distributions as dist
from pyro.optim import Adam

adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
optimizer = Adam(adam_params)


def model(latent, n):
    a = latent.a.sample_(dist.normal, Variable(torch.Tensor([10])),
                         Variable(torch.Tensor([1])))
    latent.b = named.List()
    for i, x in latent.b.irange_(10):
        x.sample_(dist.normal, a,
                  Variable(torch.Tensor([1])))


def guide(latent, n):
    latent.a.sample_(dist.normal, Variable(torch.Tensor([10])),
                     Variable(torch.Tensor([1])))
    latent.b = named.List()
    for i, x in latent.b.irange_(n):
        x.sample_(dist.normal, Variable(torch.Tensor([1])),
                  Variable(torch.Tensor([1])))


def guide2(latent, n):
    latent.b = named.List()
    for i, x in latent.b.irange_(n):
        x.sample_(dist.normal, Variable(torch.Tensor([1])),
                  Variable(torch.Tensor([1])))


def test_scoped_infer():
    trace = scoped.Importance(model, guide, num_samples=10)
    out = scoped.Marginal(trace, sites=["a"])(10)
    assert "a" in out

    def sites(latent):
        latent.b = named.List()
        for i in range(5):
            latent.b.add().set_(1)

    out = scoped.Marginal(trace, sites_fn=sites)(10)
    assert "b[2]" in out

    trace = scoped.SVI(model, guide, optim=optimizer, loss="ELBO")
    trace.step(10)


def test_condition():
    conditioned = scoped.condition(model,
                                   data={"a": Variable(torch.Tensor([10]))})

    trace = scoped.Importance(conditioned, guide2, num_samples=10)
    scoped.Marginal(trace, sites=["b[0]"])(10)

    obj = named.Object()
    conditioned(obj, 10)
    assert obj.a.data[0] == 10

    def given(latent):
        latent.b = named.List()
        for b in range(5):
            latent.b.add().set_(Variable(torch.Tensor([10])))

    conditioned = scoped.condition(model, data_fn=given)
    obj = named.Object()
    conditioned(obj, 10)
    assert obj.b[3].data[0] == 10
