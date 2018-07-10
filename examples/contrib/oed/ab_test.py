import torch
import torch.nn as nn
import numpy as np

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import EmpiricalMarginal, Importance
from pyro.contrib.oed.eig import ContinuousEIG, naiveRainforth


###################################################
# Inspired by Bayesian regression example
###################################################

# Set up regression model dimensions
N = 100  # number of participants
p_treatments = 2 # number of treatment groups
p = p_treatments  # number of features

softplus = torch.nn.Softplus()


def model(design):
    # Create normal priors over the parameters
    design_shape = design.size()
    loc = torch.zeros(*design_shape[:-2], 1, design_shape[-1])
    scale = torch.Tensor([1, .5])
    w_prior = dist.Normal(loc, scale).independent(1)
    w = pyro.sample('w', w_prior).transpose(-1, -2)

    with pyro.iarange("map", N, dim=-2):
        # run the regressor forward conditioned on inputs
        prediction_mean = torch.matmul(design, w).squeeze(-1)
        y = pyro.sample("y", dist.Normal(prediction_mean, 1).independent(1))


def guide(design):
    design_shape = design.size()
    # define our variational parameters
    w_loc = torch.zeros(*design_shape[:-2], 1, design_shape[-1])
    # note that we initialize our scales to be pretty narrow
    w_sig = -3*torch.ones(*design_shape[:-2], 1, design_shape[-1])
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_scale_weight", w_sig))
    # guide distributions for w 
    w_dist = dist.Normal(mw_param, sw_param).independent(1)
    w = pyro.sample('w', w_dist).transpose(-1, -2)


def design_to_matrix(design):
    n, p = int(torch.sum(design)), int(design.size()[0])
    X = torch.zeros(n, p)
    t = 0
    for col, i in enumerate(design):
        i = int(i)
        if i > 0:
            X[t:t+i, col] = 1.
        t += i
    return X


if __name__ == '__main__':

    ns = range(0, N, 5)
    designs = [design_to_matrix(torch.Tensor([n1, N-n1])) for n1 in ns]
    X = torch.stack(designs)

    rainforth = naiveRainforth(model, X, observation_labels="y", M=100, N=2000)
    est = ContinuousEIG(model, guide, X, vi=True)
    true = ContinuousEIG(model, guide, X, vi=False)
    

    print(est)
    print(true)
    print(rainforth)

    import matplotlib.pyplot as plt
    ns = np.array(ns)
    est = np.array(est.detach())
    true = np.array(true.detach())
    plt.scatter(ns, est)
    plt.scatter(ns, true, color='r')
    plt.scatter(ns, rainforth, color='g')
    plt.show()




















