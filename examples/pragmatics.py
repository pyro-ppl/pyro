import torch
from torch.autograd import Variable
import torch.nn as nn

import pyro
import pyro.optim
from pyro.infer import Marginal, Search, KL_QP


def world_prior():
    raise NotImplementedError("world prior is missing")


def utterance_prior():
    raise NotImplementedError("utterance prior is missing")


def dist_pow(fn, alpha):
    raise NotImplementedError("dist_pow is missing")


def L1(utt,m):
    def L1_unorm(utt, m):
        w = world_prior()
        pyro.observe("s", S1(w,m), utt)
        return w

    return Marginal(Search(L1_unorm))


def S1(w,m):
    def S1_unorm(w,m):
        u = utterance_prior()
        pyro.observe("l", L0(u,m), w)
        return u

    alpha = 2.5
    return Marginal(dist_pow(Search(S1_unorm),alpha))


def L0(utt, meaning):
    def L0_unorm(utt, meaning):
        w = world_prior()
        pyro.observe("m", Bernoulli(meaning(w,utt)), 1)
        return w

    return Marginal(Search(L0_unorm))


meaning = nn.Sequential(nn.Linear(dwu, dh), nn.ReLU(), nn.Linear(dh, 1))
    
def learner(data):
    m = pyro.module("semantics", meaning)
    pyro.map_data(data, lambda i,u,w: pyro.observe(i,S1(w,m),u))
    

stepper = KL_QP(learner, lambda d: None, pyro.optim.adam)
for i in range(1000):
    stepper(data)

print(L1("send a car!", meaning))
