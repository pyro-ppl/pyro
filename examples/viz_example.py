from pdb import set_trace as bb
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn as nn

import pyro
import pyro.distributions
from pyro.distributions import diagnormal
from pyro.distributions import gamma
from pyro.distributions import DiagNormal
import pyro.poutine as poutine
import networkx

pt_lin = nn.Linear(1,1)

def model():
    mu1 = pyro.param("mu1", Variable(torch.ones(1)))
    mu2 = pyro.param("mu2", Variable(torch.zeros(1)))
    mu3 = pyro.param("mu3", Variable(torch.zeros(1)))
    z = pyro.sample("z", diagnormal, torch.abs(mu1+mu2),
                                    Variable(torch.ones(1)), reparameterized=False)
    latent1 = pyro.sample("latent1", diagnormal, mu1*mu1,
                                    Variable(torch.ones(1)), reparameterized=False)
    latent2 = pyro.sample("latent2", diagnormal, mu2,
                                    Variable(torch.ones(1)), reparameterized=False)
    latent3 = pyro.sample("latent3", diagnormal, latent2+mu3,
                                    Variable(torch.ones(1)), reparameterized=False)
    lin = pyro.module("linear", pt_lin)
    latent = lin(latent1.view(1,1)).view(1) + lin(latent3.view(1,1)).view(1) + z
    x_dist = DiagNormal(latent, Variable(torch.ones(1)))
    x = pyro.observe("obs", x_dist, Variable(torch.ones(1)))
    return latent

mymodel = poutine.tracegraph(model, graph_output='skinnygraph',
                      skip_creators = True, include_intermediates = False)
tracegraph = mymodel()
print "nodes:\n", tracegraph.get_nodes()
print "non_reparam_nodes:\n", tracegraph.get_nonreparam_stochastic_nodes()
print "stocha:", tracegraph.stochastic_nodes
print "reparam:", tracegraph.reparameterized_nodes
print "paramnodes:", tracegraph.param_nodes
print "direct children:", tracegraph.get_direct_stochastic_children_of_parameters()

for node in tracegraph.get_nodes():
    print "%s ancestors:" % node, tracegraph.get_ancestors(node)
    print "%s predecessors=parents:" % node, tracegraph.get_parents(node)
    print "%s successors=children:" % node, tracegraph.get_children(node)
    print "%s descendants:" % node, tracegraph.get_descendants(node)

print "tracegraph trace:\n", tracegraph.get_trace()

mymodel = poutine.tracegraph(model, graph_output='fullgraph',
                      skip_creators = False, include_intermediates = True)
tracegraph = mymodel()

mymodel = poutine.tracegraph(model, graph_output='graph.nofuncs',
                      skip_creators = True, include_intermediates = True)
tracegraph = mymodel()
