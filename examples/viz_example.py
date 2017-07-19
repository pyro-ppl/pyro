from pdb import set_trace as bb
import numpy as np
import torch
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal
import pyro.poutine as poutine


def model():
    mu1 = pyro.param("mu1", Variable(torch.zeros(1)))
    mu2 = pyro.param("mu2", Variable(torch.zeros(1)))
    latent1 = pyro.sample("latent1",
                         DiagNormal(mu1,
                                    Variable(torch.ones(1))))
    latent2 = pyro.sample("latent2",
                         DiagNormal(mu2,
                                    Variable(torch.ones(1))))
    latent = latent1 + latent2
    x_dist = DiagNormal(latent, Variable(torch.ones(1)))
    x = pyro.observe("obs", x_dist, Variable(torch.ones(1)))
    return latent

mymodel = poutine.viz(model, output_file='viz.output')
mymodel()
