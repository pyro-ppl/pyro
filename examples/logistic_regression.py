import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import normalize
from pdb import set_trace as bb

from torch.autograd import Variable
import pandas as pd

import pyro
from pyro import poutine
from pyro.distributions import DiagNormal, Bernoulli
from pyro.infer.kl_qp import KL_QP
import torchvision.datasets as dset

"""
Bayesian Logistic Regression
"""

# use covtype dataset
# fname = "data/covtype/covtype.data"
# print("loading covtype data set...")
# with open(fname, "r+") as f:
#     content = f.read()
# df = pd.read_csv(fname, header=None)
# print("...done")

# generate toy dataset
def build_toy_dataset(N, noise_std=0.1):
  D = 1
  X = np.linspace(-6, 6, num=N)
  y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
  y[y < 0.5] = 0
  y[y >= 0.5] = 1
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D))
  y = y.reshape((N, 1))
  X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
  return torch.cat((X,y), 1)
  return Variable(torch.Tensor(X)), Variable(torch.Tensor(y))

sigmoid = torch.nn.Sigmoid()
softplus = torch.nn.Softplus()

# N = 581012 # data
# D = 54 # features
N = 10000 # toy data
D = 1
batch_size = 256

"""
Custom prior we specify. This can be anything, but for this example
we use DiagNormal
"""
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

def model(data):
    x_data = data[:,:-1]
    y_data = data[:,-1]
    mu = Variable(torch.zeros(D, 1))
    sigma = Variable(torch.ones(D, 1))
    bias_mu = Variable(torch.zeros(1))
    bias_sigma = Variable(10.0*torch.ones(1))
    w_prior = DiagNormalPrior(mu, sigma)
    b_prior = DiagNormalPrior(bias_mu, bias_sigma)
    priors = {'weight': w_prior, 'bias': b_prior}
    lifted_fn = poutine.lift(log_reg, priors)
    latent = lifted_fn(x_data)
    pyro.observe("obs", Bernoulli(latent), y_data.unsqueeze(1))


def guide(data):
    x_data = data[:,:-1]
    w_mu = Variable(torch.randn(D, 1), requires_grad=True)
    w_log_sig = Variable(-3.0*torch.ones(D,1) + 0.05*torch.randn(D, 1), requires_grad=True)
    b_mu = Variable(torch.randn(1), requires_grad=True)
    b_log_sig = Variable(-3.0*torch.ones(1) + 0.05 * torch.randn(1), requires_grad=True)
    mw_param = pyro.param("guide_mean_weight", w_mu)
    sw_param = pyro.param("guide_sigma_weight", w_log_sig)
    mb_param  = pyro.param("guide_mean_bias", b_mu)
    sb_param = pyro.param("guide_sigma_bias", b_log_sig)
    sw_param = torch.exp(sw_param)
    sb_param = torch.exp(sb_param)
    w_prior = DiagNormalPrior(mw_param, sw_param)
    b_prior = DiagNormalPrior(mb_param, sb_param)
    priors = {'weight': w_prior, 'bias': b_prior}
    lifted_fn = poutine.lift(log_reg, priors)
    lifted_fn(x_data)

# adam_params = {"lr": 0.00001, "betas": (0.95, 0.999)}
lr = {"lr": 0.001}
adam_optim = pyro.optim(torch.optim.Adam, lr)
sgd_optim = pyro.optim(torch.optim.SGD, lr)

# dat = build_toy_dataset(N)
# x = df.as_matrix(columns=range(D))
# y = df.as_matrix(columns=[D])
# raw_data = Variable(torch.Tensor(df.as_matrix().tolist()))
# data = normalize(raw_data, 2, dim=1)
# x_norm = normalize(Variable(torch.Tensor(x.tolist())), 2, dim=1)
# y = Variable(torch.Tensor(y.tolist()))
# data = torch.cat((x_norm, y), 1)
data = build_toy_dataset(N)

def inspect_post_params():
    print pyro.get_param_store._params
    mw_param = pyro.param("guide_mean_weight")
    sw_param = pyro.param("guide_sigma_weight")
    mb_param  = pyro.param("guide_mean_bias")
    sb_param = pyro.param("guide_sigma__bias")
    tuples = [("weight mean", mw_param), ("weight sigma", sw_param),\
              ("mean bias", mb_param), ("bias sigma", sb_param)]
    return iter(tuples)

nr_samples = 50
nr_epochs = 1000
all_batches = np.arange(0, N, batch_size)
# take care of bad index
if all_batches[-1] != N:
    all_batches = list(all_batches) + [N]

grad_step = KL_QP(model, guide, adam_optim)

# apply it to minibatches of data by hand:
for j in range(nr_epochs):
    epoch_loss = 0.0
    for ix, batch_start in enumerate(all_batches[:-1]):
        batch_end = all_batches[ix + 1]
        batch_data = data[batch_start: batch_end]
        epoch_loss += grad_step.step(batch_data)
        print pyro.get_param_store()._params
    print("epoch avg loss {}".format(epoch_loss/float(N)))
