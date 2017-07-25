from pdb import set_trace as bb
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn as nn

import pyro
from pyro.distributions import DiagNormal
import pyro.poutine as poutine
import networkx

pt_lin = nn.Linear(1,1)

def model():
    mu1 = pyro.param("mu1", Variable(torch.ones(1)))
    mu2 = pyro.param("mu2", Variable(torch.zeros(1)))
    #z = pyro.sample("z",  DiagNormal(torch.abs(mu1+mu2),
    #                                Variable(torch.ones(1)), reparameterized=False))
    z = diagnormal(torch.abs(mu1+mu2), Variable(torch.ones(1)), reparameterized=False)
    latent1 = pyro.sample("latent1",
                         DiagNormal(mu1*mu1,
                                    Variable(torch.ones(1)), reparameterized=False))
    latent2 = pyro.sample("latent2",
                         DiagNormal(mu2,
                                    Variable(torch.ones(1))))
    lin = pyro.module("linear", pt_lin)
    latent = lin(latent1.view(1,1)).view(1) + lin(latent2.view(1,1)).view(1) + z
    x_dist = DiagNormal(latent, Variable(torch.ones(1)))
    x = pyro.observe("obs", x_dist, Variable(torch.ones(1)))
    return latent

mymodel = poutine.viz(model, output_file='skinnygraph',
                      skip_creators = True, include_intermediates = False)
graph = mymodel()
print "nodes:\n", graph.get_nodes()
print "non_reparam_nodes:\n", graph.get_nonreparam_stochastic_nodes()
print "stocha:", graph.stochastic_nodes
print "reparam:", graph.reparameterized_nodes
print "paramnodes:", graph.param_nodes
print "direct children:", graph.get_direct_stochastic_children_of_parameters()

for node in graph.get_nodes():
    print "%s ancestors:" % node, graph.get_ancestors(node)
    print "%s predecessors=parents:" % node, graph.get_parents(node)
    print "%s successors=children:" % node, graph.get_children(node)
    print "%s descendants:" % node, graph.get_descendants(node)

mymodel = poutine.viz(model, output_file='fullgraph',
                      skip_creators = False, include_intermediates = True)
graph = mymodel()

mymodel = poutine.viz(model, output_file='graph.nofuncs',
                      skip_creators = True, include_intermediates = True)
graph = mymodel()
