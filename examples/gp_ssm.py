#!/usr/bin/env python

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyro
from pyro.distributions import DiagNormal, Bernoulli, Categorical, LogNormal, Normal
from pyro.infer.kl_qp import KL_QP


def transition_kernel(prev_state, delta=1.0, **gp_hypers):

    sigvar = gp_hypers.get("sigvar", 1.0)
    lambda_ = gp_hypers.get("lambda_", 1.0)
    logq = np.log(2*sigvar) + np.log(4) + 3*np.log(lambda_) - np.log(2.0)
    q = np.exp(logq)

    if prev_state is None:
        mu = np.ones(2) # assume zero prior mean
        Q = np.array([[sigvar, 0.0],
                      [0.0, lambda_**2*sigvar]])
    else:
        Phi = np.array([[np.exp(-lambda_*delta)*(1 + lambda_*delta), delta*np.exp(-lambda_*delta)],
                        [-(lambda_**2)*delta*np.exp(-lambda_*delta), np.exp(-lambda_*delta)*(1 - lambda_*delta)]])
        
        Q = np.zeros_like(Phi)
        Q[0, 0] = 1/(4*lambda_**3) - (4*delta**2*lambda_**2 + 4*delta*lambda_ + 2)/(8*lambda_**3*np.exp(2*delta*lambda_))
        Q[0, 1] = delta**2/(2*np.exp(2*delta*lambda_))
        Q[1, 0] = Q[0, 1]
        Q[1, 1] = 1/(4*lambda_) - (2*delta**2*lambda_**2 - 2*delta*lambda_ + 1)/(4*lambda_*np.exp(2*delta*lambda_))
        Q *= q
        mu = np.dot(Phi, prev_state)

    return mu, Q


def model(data):

    # TODO intro GP hypers to autograd, need to discuss best way to do this
    #lambda_ = pyro.param("lambda_", Variable(torch.ones(1), requires_grad=True))
    #sigvar = pyro.param("sigvar", Variable(torch.ones(1), requires_grad=True))
    noise_var = Variable(torch.ones(1), requires_grad=False)

    
    for t in xrange(data.size(0)):
        if t == 0:
            mu, V = transition_kernel(None)
        else:
            delta = data[t, 0] - data[t-1, 0]
            assert delta >= 0.0, "inputs must be sorted"
            mu, V = transition_kernel(z, delta=delta)
        z = pyro.sample("z_%i" % t, Normal(Variable(torch.from_numpy(mu), requires_grad=False),
                                           Variable(torch.from_numpy(V), requires_grad=False)))
        pyro.observe("timestep_%i" % t,
                     Normal(z[0], noise_var), data[t, 1]) # Can have arbirary datatypes here! 

def guide(data):

    lstm1 = nn.LSTMCell(2, 51).double() # vis_dim, hidden_dim
    lstm2 = nn.LSTMCell(51, 1).double()

    # a world without a batch index is unconceivable to deep learning folks
    h_t = Variable(torch.zeros(1, 51).double(), requires_grad=False)
    c_t = Variable(torch.zeros(1, 51).double(), requires_grad=False)
    h_t2 = Variable(torch.zeros(1, 1).double(), requires_grad=False)
    c_t2 = Variable(torch.zeros(1, 1).double(), requires_grad=False)

    for t in xrange(data.size(0)):
        pyro.param("z_var_%i" % t, Variable(torch.ones(1), requires_grad=True))
    
    lstm_outputs = []
    for t, data_t in enumerate(data.chunk(data.size(0), dim=0)):
        h_t, c_t = lstm1(data_t, (h_t, c_t))
        h_t2, c_t2 = lstm2(c_t, (h_t2, c_t2))
        pyro.sample("z_%i" % t, LogNormal(c_t2[0, 0].float(), pyro.param("z_var_%i" % t)))
        lstm_outputs.append(c_t2)

    return lstm_outputs


if __name__ == '__main__':
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    time_series = Variable(torch.from_numpy(np.concatenate([np.ones(len(data[1])).astype(np.float32)[:, None], # input diffs
                                                            data[1][:, None]], axis=1)), requires_grad=False)
    print time_series.size()
    # build the model

    optim_fct = pyro.optim(torch.optim.Adam, {'lr': .0001})
    # TODO this is a batch size of 1, how do you do minibatches
    grad_step = KL_QP(model, guide, optim_fct)

    nr_epochs = 20
    for ii in xrange(10):
        loss = grad_step(time_series)
        print "loss: %.3f" % loss
