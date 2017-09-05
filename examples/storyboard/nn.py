from __future__ import print_function
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pyro
from pyro import poutine
from pyro.distributions import DiagNormal, Categorical
from pyro.infer.kl_qp import KL_QP
#To start, let's just write a non-bayesian neural net, with supervised training:

# simple nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))

def prior():
    mu = Variable(torch.zeros(2))
    sigma = Variable(torch.ones(2))
    return pyro.sample("sample", DiagNormal(mu, sigma))

def model(data):
    model = pyro.random_module(module, prior)
    pyro.observe()
    pass

def guide(data):
    pass


#now let's make it bayesian. we'll need the prior fn:
def prior(name, shape):
    for p in shape:
        yield pyro.sample(name+uuid, DiagNormal(p.size()))

#note that in the corresponding guide (for VI) we'd want to add (implicitly or explicitly) params to the prior dist.
#here's a sketch of a bayesian nn with a layer-wise independent posterior:

def prior(name, shape):
    for p in shape:
        yield pyro.sample(name+uuid, Gaussian(p.size()))

def posterior(name, shape):
    for p in shape:
        weight_dist = dist.DiagNormal(pyro.param(name+"param"+uuid, p.size()))
        yield pyro.sample(name+uuid, weight_dist)

def model(data):
    classifier_dist = pyro.random_module("classifier", classify, prior) #make the module into an implicit distribution on nets
    nn = classifier_dist() #sample a random net
    map_data(data, lambda i, d: pyro.observe("obs"+i, Categorical(nn.forward(d.data)), d.cll))

def guide(data):
    classifier_dist = pyro.random_module("classifier", classify, posterior) #make the module into an implicit distribution on nets
    nn = classifier_dist() #sample a random net


def posterior(name, shape):
    update = pyro.module("update", update)
    predict = pyro.module("predict", predict)
    hidden = Variable(nn.ones(10))
    for p in shape:
        weight_dist = Gaussian(predict.forward(hidden))
        weights = pyro.sample(name+uuid, weight_dist)
        hidden = update.forward(nn.concat(weights, hidden))
        yield weights
