import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize  # noqa: F401

import pyro
from pyro.distributions import Bernoulli, Normal  # noqa: F401
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


"""
Bayesian Regression
Learning a function of the form:
    y = wx + b
"""


# generate toy dataset
def build_linear_dataset(N, p, noise_std=0.01):
    X = np.random.rand(N, p)
    # use random integer weights from [0, 7]
    w = np.random.randint(4, size=p)
    print('w = {}'.format(w))
    # set b = 1
    y = np.matmul(X, w) + np.repeat(1, N) + np.random.normal(0, noise_std, size=N)
    y = y.reshape(N, 1)
    X, y = torch.tensor(X), torch.tensor(y)
    data = torch.cat((X, y), 1)
    assert data.shape == (N, p + 1)
    return data


# NN with one linear layer
class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        # x * w + b
        return self.linear(x)


N = 100  # size of toy data
p = 2  # number of features

softplus = nn.Softplus()
regression_model = RegressionModel(p)


def model(data):
    # Create unit normal priors over the parameters
    loc = data.new_zeros(torch.Size((1, p)))
    scale = 2 * data.new_ones(torch.Size((1, p)))
    bias_loc = data.new_zeros(torch.Size((1,)))
    bias_scale = 2 * data.new_ones(torch.Size((1,)))
    w_prior = Normal(loc, scale).independent(1)
    b_prior = Normal(bias_loc, bias_scale).independent(1)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    with pyro.iarange("map", N, subsample=data):
        x_data = data[:, :-1]
        y_data = data[:, -1]
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        pyro.sample("obs", Normal(prediction_mean, 1),
                    obs=y_data)


def guide(data):
    w_loc = data.new_tensor(torch.randn(1, p))
    w_log_sig = data.new_tensor(-3.0 * torch.ones(1, p) + 0.05 * torch.randn(1, p))
    b_loc = data.new_tensor(torch.randn(1))
    b_log_sig = data.new_tensor(-3.0 * torch.ones(1) + 0.05 * torch.randn(1))
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_loc)
    sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))
    # gaussian guide distributions for w and b
    w_dist = Normal(mw_param, sw_param).independent(1)
    b_dist = Normal(mb_param, sb_param).independent(1)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
    # overloading the parameters in the module with random samples from the guide distributions
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor
    return lifted_module()


# instantiate optim and inference objects
optim = Adam({"lr": 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO())


# get array of batch indices
def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


def main(args):
    pyro.clear_param_store()
    data = build_linear_dataset(N, p)
    if args.cuda:
        # make tensors and modules CUDA
        data = data.cuda()
        softplus.cuda()
        regression_model.cuda()
    for j in range(args.num_epochs):
        if args.batch_size == N:
            # use the entire data set
            epoch_loss = svi.step(data)
        else:
            # mini batch
            epoch_loss = 0.0
            perm = torch.randperm(N) if not args.cuda else torch.randperm(N).cuda()
            # shuffle data
            data = data[perm]
            # get indices of each batch
            all_batches = get_batch_indices(N, args.batch_size)
            for ix, batch_start in enumerate(all_batches[:-1]):
                batch_end = all_batches[ix + 1]
                batch_data = data[batch_start: batch_end]
                epoch_loss += svi.step(batch_data)
        if j % 100 == 0:
            print("epoch avg loss {}".format(epoch_loss/float(N)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    parser.add_argument('-b', '--batch-size', default=N, type=int)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    main(args)
