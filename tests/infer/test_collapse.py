from __future__ import absolute_import, division, print_function

import logging
import warnings

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

logger = logging.getLogger(__name__)


def whiteboard_model():

    loc_z2 = pyro.param("loc_z2", torch.randn(3))
    loc_x2 = pyro.param("loc_x2", torch.randn(3))
    scale_x3 = pyro.param("scale_x3", torch.randn(3),
                          constraint=constraints.positive)

    e = pyro.sample("e", dist.Categorical(torch.ones(3)))

    z1 = pyro.sample("z1", dist.Normal(0., 1.))
    z2 = pyro.sample("z2", dist.Normal(loc_z2[e], 1.0))

    x1 = pyro.sample("x1", dist.Normal(z2, 1.), obs=torch.tensor(1.7))
    x2 = pyro.sample("x2", dist.Normal(loc_x2[e], 1.), obs=torch.tensor(2.))
    x3 = pyro.sample("x3", dist.Normal(z1, scale_x3[e]), obs=torch.tensor(1.6))
    x4 = pyro.sample("x4", dist.Normal(z1, 1.), obs=torch.tensor(1.5))
    
    return x1, x2, x3, x4
