import numpy as np
import torch
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal
import pyro.poutine as poutine


def model():
    latent = pyro.sample("latent",
                         DiagNormal(Variable(torch.zeros(1)),
                                    5 * Variable(torch.ones(1))))
    x_dist = DiagNormal(latent, Variable(torch.ones(1)))
    x = pyro.observe("obs", x_dist, Variable(torch.ones(1)))
    return latent


def guide():
    latent = pyro.sample("latent",
                         DiagNormal(Variable(torch.zeros(1)),
                                    5 * Variable(torch.ones(1))))
    #x_dist = DiagNormal(latent, Variable(torch.ones(1)))
    return latent

def eq(x, y, prec=1e-10):
    vv = (torch.norm(x-y).data[0] < prec)
    print(vv)
    if not vv:
        print(x, y)
    return vv


model_trace = poutine.trace(model)()
guide_trace = poutine.trace(guide)()

model_trace_replay = poutine.replay(poutine.trace(model), guide_trace)()
model_replay_trace = poutine.trace(poutine.replay(model, guide_trace))()
model_replay_ret = poutine.replay(model, guide_trace)()

assert(eq(model_trace_replay["_RETURN"]["value"], model_replay_ret))

assert(eq(model_replay_ret, guide_trace["latent"]["value"]))

assert(eq(model_replay_trace["latent"]["value"],
          guide_trace["latent"]["value"]))

assert(not eq(model_replay_trace["latent"]["value"],
              model_trace_replay["latent"]["value"]))

assert(not eq(model_trace["latent"]["value"], guide_trace["latent"]["value"]))
