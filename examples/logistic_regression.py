import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.nn.functional import normalize  # noqa: F401

from torch.autograd import Variable
# import pandas as pd  # noqa: F401

import pyro
from pyro import poutine
from pyro.distributions import DiagNormal, Bernoulli
from pyro.infer import SVI
from pyro.optim import Adam

"""
Bayesian Logistic Regression
"""

# use UCI dataset
# fname = "data/covtype.data"
# print("loading covtype data set...")
# with open(fname, "r+") as f:
#     content = f.read()
# df = pd.read_csv(fname, header=None)
# print("...done")

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

# N = 581012  # data
# D = 54  # features
N = 1000  # toy data
D = 1
batch_size = 256


# Custom prior we specify. This can be anything, but for this example
# we use DiagNormal
class DiagNormalPrior(pyro.distributions.Distribution):

    def __init__(self, mu, sigma, *args, **kwargs):
        self.mu = mu
        self.sigma = sigma
        self.dist = DiagNormal(self.mu, self.sigma)
        super(DiagNormalPrior, self).__init__(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.dist()

    def log_pdf(self, x, *args, **kwargs):
        return self.dist.log_pdf(x)


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

    pyro.map_data("map", data, lambda i, d: observe(d), batch_size=10)


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
    pyro.map_data("map", x_data, lambda i, d: lifted_fn(d), batch_size=10)


adam = Adam({"lr": 0.01})
svi = SVI(model, guide, adam, loss="ELBO")

# For UCI dataset
# x = df.as_matrix(columns=range(D))
# y = df.as_matrix(columns=[D])
# raw_data = Variable(torch.Tensor(df.as_matrix().tolist()))
# data = normalize(raw_data, 2, dim=1)
# x_norm = normalize(Variable(torch.Tensor(x.tolist())), 2, dim=1)
# y = Variable(torch.Tensor(y.tolist()))
# data = torch.cat((x_norm, y), 1)

# For toy dataset
data = build_linear_dataset(N)

all_batches = np.arange(0, N, batch_size)
# take care of bad index
if all_batches[-1] != N:
    all_batches = list(all_batches) + [N]


def main():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', nargs='?', default=10000, type=int)
    args = parser.parse_args()
    for j in range(args.num_epochs):
        epoch_loss = svi.step(data)
#         # mini batch
#         epoch_loss = 0.0
#         for ix, batch_start in enumerate(all_batches[:-1]):
#             batch_end = all_batches[ix + 1]
#             batch_data = data[batch_start: batch_end]
#             epoch_loss += svi.step(batch_data)
        if j % 200 == 0:
            print pyro.get_param_store()._params
            print("epoch avg loss {}".format(epoch_loss/float(N)))


if __name__ == '__main__':
    main()
