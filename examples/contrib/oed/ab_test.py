import torch
import torch.nn as nn
import numpy as np

import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance
from pyro.contrib.oed.eig import ContinuousEIG


###################################################
# Inspired by Bayesian regression example
###################################################


# NN with one linear layer
class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        # Won't need a bias with our one-hot setup
        self.linear = nn.Linear(p, 1, bias=False)

    def forward(self, x):
        # x * w + b
        return self.linear(x)

# Set up regression model dimensions
N = 100  # number of participants
p_treatments = 2 # number of treatment groups
p = p_treatments  # number of features

softplus = torch.nn.Softplus()
regression_model = RegressionModel(p)


def model(design):
    # Create unit normal priors over the parameters
    loc = torch.zeros(1, p)
    scale = torch.Tensor([1, .1])
    w_prior = dist.Normal(loc, scale).independent(1)
    priors = {'linear.weight': w_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    X = design_to_matrix(design)

    with pyro.iarange("map", N, subsample=X):
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(X).squeeze(-1)
        pyro.sample("y", dist.Normal(prediction_mean, 1))


def guide(design):
    # define our variational parameters
    w_loc = torch.zeros(1, p)
    # note that we initialize our scales to be pretty narrow
    w_sig = -3*torch.ones(1, p)
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_scale_weight", w_sig))
    # guide distributions for w 
    w_dist = dist.Normal(mw_param, sw_param).independent(1)
    dists = {'linear.weight': w_dist}
    # overload the parameters in the module with random samples
    # from the guide distributions
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()


def design_to_matrix(design):
    n, p = int(torch.sum(design)), int(design.size()[0])
    X = torch.zeros(n, p)
    t = 0
    for col, i in enumerate(design):
        i = int(i)
        X[t:t+i, col] = 1.
        t += i
    return X


if __name__ == '__main__':
    ns = [40, 45, 50, 55, 60, 65]
    true = []
    est = []
    for n1 in ns:
        print(n1)
        point = torch.Tensor([n1, N - n1])
        est.append(ContinuousEIG(model, guide, point, vi=True))
        true.append(ContinuousEIG(model, guide, point, vi=False))

    print(est)
    print(true)

    import matplotlib.pyplot as plt
    ns = np.array(ns)
    est = np.array(est)
    true = np.array(true)
    plt.scatter(ns, est)
    plt.scatter(ns, true, color='r')
    plt.show()




















