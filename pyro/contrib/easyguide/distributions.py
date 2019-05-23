from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist


class EasyDistribution(object):
    pass


class EasyDelta(EasyDistribution):
    pass


class EasyNormal(EasyDistribution):
    def __init__(self, loc="auto", scale="auto", init_fn=None):
        self.loc = loc
        self.scale = scale
        self.init_fn = init_fn

    def __call__(self, sites):
        pass
