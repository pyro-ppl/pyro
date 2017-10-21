import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.nn.functional import normalize  # noqa: F401
from pdb import set_trace as bb

from torch.autograd import Variable

import pyro
from pyro import poutine
from pyro.distributions import DiagNormal, Bernoulli  # noqa: F401
from pyro.infer import SVI
from pyro.optim import Adam

"""
Bayesian Regression
Learning a function of the form:
    y = wx + b
"""

# generate toy dataset
def build_linear_dataset(N, p, noise_std=0.1):
    X = np.linspace(-6, 6, num=N)
    y = 3 * X + 1 + np.random.normal(0, noise_std, size=N)
    X = X.reshape((N, p))
    y = y.reshape((N, 1))
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X, y), 1)


# NN with one linear layer
class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        # x * w + b
        return self.linear(x)


N = 100  # size of toy data
p = 1  # number of features

softplus = nn.Softplus()
regression_model = RegressionModel(p)


def model(data):
    mu = Variable(torch.zeros(p, 1))
    sigma = Variable(torch.ones(p, 1))
    bias_mu = Variable(torch.zeros(1))
    bias_sigma = Variable(torch.ones(1))
    w_prior = DiagNormal(mu, sigma)
    b_prior = DiagNormal(bias_mu, bias_sigma)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a nn
    lifted_nn = lifted_module()

    with pyro.iarange("map", N, subsample=data):
        x_data = data[:, :-1]
        y_data = data[:, -1]
        # run the nn with the data
        latent = lifted_nn(x_data).squeeze()
        pyro.observe("obs", DiagNormal(latent, Variable(torch.ones(data.size(0)))), y_data.squeeze())


def guide(data):
    w_mu = Variable(torch.randn(p, 1), requires_grad=True)
    w_log_sig = Variable(-3.0 * torch.ones(p, 1) + 0.05 * torch.randn(p, 1), requires_grad=True)
    b_mu = Variable(torch.randn(1), requires_grad=True)
    b_log_sig = Variable(-3.0 * torch.ones(1) + 0.05 * torch.randn(1), requires_grad=True)
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_mu)
    sw_param = softplus(pyro.param("guide_sigma_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_mu)
    sb_param = softplus(pyro.param("guide_sigma_bias", b_log_sig))
    w_prior = DiagNormal(mw_param, sw_param)
    b_prior = DiagNormal(mb_param, sb_param)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    # overloading the parameters in the module with random samples from the prior
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a nn
    lifted_nn = lifted_module()


# instantiate optim and inference objects
optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss="ELBO")

# get array of batch indices
def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


def main():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    parser.add_argument('-b', '--batch-size', default=N, type=int)
    args = parser.parse_args()
    data = build_linear_dataset(N, p)
    for j in range(args.num_epochs):
        if args.batch_size == N:
            epoch_loss = svi.step(data)
        else:
            # mini batch
            epoch_loss = 0.0
            # shuffle data
            data = data[torch.randperm(N)]
            # get indices of each batch
            all_batches = get_batch_indices(N, args.batch_size)
            for ix, batch_start in enumerate(all_batches[:-1]):
                batch_end = all_batches[ix + 1]
                batch_data = data[batch_start: batch_end]
                epoch_loss += svi.step(batch_data)
        if j % 100 == 0:
            print("epoch avg loss {}".format(epoch_loss/float(N)))


if __name__ == '__main__':
    main()
