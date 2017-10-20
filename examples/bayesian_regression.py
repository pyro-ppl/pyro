import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.nn.functional import normalize  # noqa: F401

from torch.autograd import Variable

import pyro
from pyro import poutine
from pyro.distributions import DiagNormal, Bernoulli  # noqa: F401
from pyro.infer import SVI
from pyro.optim import Adam

"""
Bayesian Regression
"""


# generate toy dataset
def build_linear_dataset(N, noise_std=0.1):
    D = 1
    X = np.linspace(-6, 6, num=N)
    y = 3 * X + 1 + np.random.normal(0, noise_std, size=N)
    X = X.reshape((N, D))
    y = y.reshape((N, 1))
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X, y), 1)


def build_logistic_dataset(N, noise_std=0.1):
    D = 1
    X = np.linspace(-6, 6, num=N)
    y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    X = (X - 4.0) / 4.0
    X = X.reshape((N, D))
    y = y.reshape((N, 1))
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X, y), 1)


sigmoid = nn.Sigmoid()
softplus = nn.Softplus()

N = 1000  # size of toy data
D = 1  # number of features
batch_size = 256  # batch size


def log_reg(x_data):
    p_w = pyro.param("weight", Variable(torch.zeros(D, 1), requires_grad=True))
    p_b = pyro.param("bias", Variable(torch.ones(1), requires_grad=True))
    reg = torch.addmm(p_b, x_data, p_w)
    latent = sigmoid(reg)
    return latent


def lin_reg(x_data):
    p_w = pyro.param("weight", Variable(torch.zeros(1), requires_grad=True))
    p_b = pyro.param("bias", Variable(torch.ones(1), requires_grad=True))
    latent = p_w.squeeze() * x_data.squeeze() + p_b
    return latent


def model(data):
    mu = Variable(torch.zeros(D, 1))
    sigma = Variable(torch.ones(D, 1))
    bias_mu = Variable(torch.zeros(1))
    bias_sigma = Variable(torch.ones(1))
    w_prior = DiagNormal(mu, sigma)
    b_prior = DiagNormal(bias_mu, bias_sigma)
    priors = {'weight': w_prior, 'bias': b_prior}
    lifted_fn = poutine.lift(lin_reg, priors)

    def observe(data):
        x_data = data[:, :-1]
        y_data = data[:, -1]
        latent = lifted_fn(x_data)
        pyro.observe("obs", DiagNormal(latent, Variable(torch.ones(10))), y_data.squeeze())

    batch_size = 10
    data = data[torch.randperm(data.size(0))]
    subsample = data[0:batch_size]
    with pyro.iarange("map", N, subsample=subsample) as batch:
        observe(data.index_select(0, batch))


def guide(data):
    x_data = data[:, :-1]
    w_mu = Variable(torch.randn(D, 1), requires_grad=True)
    w_log_sig = Variable(-3.0 * torch.ones(D, 1) + 0.05 * torch.randn(D, 1), requires_grad=True)
    b_mu = Variable(torch.randn(1), requires_grad=True)
    b_log_sig = Variable(-3.0 * torch.ones(1) + 0.05 * torch.randn(1), requires_grad=True)
    mw_param = pyro.param("guide_mean_weight", w_mu)
    sw_param = softplus(pyro.param("guide_sigma_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_mu)
    sb_param = softplus(pyro.param("guide_sigma_bias", b_log_sig))
    w_prior = DiagNormal(mw_param, sw_param)
    b_prior = DiagNormal(mb_param, sb_param)
    priors = {'weight': w_prior, 'bias': b_prior}
    lifted_fn = poutine.lift(lin_reg, priors)
    batch_size = 10
    x_data = x_data[torch.randperm(x_data.size(0))]
    subsample = x_data[0:batch_size]
    with pyro.iarange("map", N, subsample=subsample) as batch:
        lifted_fn(x_data.index_select(0, batch))


adam = Adam({"lr": 0.01})
svi = SVI(model, guide, adam, loss="ELBO")


def load_data(reg_type):
    if reg_type == 'linear':
        return build_linear_dataset(N)
    elif reg_type == 'logistic':
        return build_logistic_dataset(N)
    raise ValueError('data set type should be "logistic" or "linear".')


# get batch indices
all_batches = np.arange(0, N, batch_size)
if all_batches[-1] != N:
    all_batches = list(all_batches) + [N]


def main():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    parser.add_argument('-b', '--batch', default=False, type=bool)
    parser.add_argument('-t', '--regression-type', default='linear', type=str)
    args = parser.parse_args()
    data = load_data(args.regression_type)
    for j in range(args.num_epochs):
        if not args.batch:
            epoch_loss = svi.step(data)
        else:
            # mini batch
            epoch_loss = 0.0
            # shuffle data
            data = data[torch.randperm(N)]
            for ix, batch_start in enumerate(all_batches[:-1]):
                batch_end = all_batches[ix + 1]
                batch_data = data[batch_start: batch_end]
                epoch_loss += svi.step(batch_data)
        if j % 100 == 0:
            print("epoch avg loss {}".format(epoch_loss/float(N)))


if __name__ == '__main__':
    main()
