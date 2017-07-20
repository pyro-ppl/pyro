from pdb import set_trace as bb
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn as nn

import pyro
from pyro.distributions import DiagNormal
import pyro.poutine as poutine

pt_lin = nn.Linear(1,1)

def model():
    mu1 = pyro.param("mu1", Variable(torch.zeros(1)))
    mu2 = pyro.param("mu2", Variable(torch.zeros(1)))
    latent1 = pyro.sample("latent1",
                         DiagNormal(mu1,
                                    Variable(torch.ones(1))))
    latent2 = pyro.sample("latent2",
                         DiagNormal(mu2,
                                    Variable(torch.ones(1))))
    lin = pyro.module("linear", pt_lin)
    latent = lin(latent1.view(1,1)).view(1) + lin(latent2.view(1,1)).view(1)
    x_dist = DiagNormal(latent, Variable(torch.ones(1)))
    x = pyro.observe("obs", x_dist, Variable(torch.ones(1)))
    return latent

mymodel = poutine.viz(model, output_file='skinnygraph',
                      skip_creators = True, include_intermediates = False)
mymodel()

mymodel = poutine.viz(model, output_file='fullgraph',
                      skip_creators = False, include_intermediates = True)
mymodel()

mymodel = poutine.viz(model, output_file='graph.nofuncs',
                      skip_creators = True, include_intermediates = True)
mymodel()
