import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import normalize

from torch.autograd import Variable
import pandas as pd

import pyro
from pyro import poutine
from pyro.distributions import DiagNormal, Categorical
from pyro.infer.kl_qp import KL_QP
import torchvision.datasets as dset

"""
Estimates w s.t. y = Xw, using Bayesian regularisation.
Replicates the result in https://arxiv.org/pdf/1310.5438.pdf
"""

# use covtype dataset
fname = "data/covtype/covtype.data"
print("loading covtype data set...")
with open(fname, "r+") as f:
    content = f.read()
df = pd.read_csv(fname, header=None)
print("...done")

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
  return Variable(torch.Tensor(X)), Variable(torch.Tensor(y))

sigmoid = torch.nn.Sigmoid()
softplus = torch.nn.Softplus()

N = 581012 # data
# D = 39 # features
D = 54 # features
batch_size = 256

def model(data):
    """
    Logistic regression model
    """
    x_data = data[:,:-1]
    y_data = data[:,-1]
    mu = Variable(torch.zeros(D, 1))
    sigma = Variable(torch.ones(D, 1))
    bias_mu = Variable(torch.zeros(1))
    bias_sigma = Variable(10.0*torch.ones(1))
    p_w = pyro.sample("weight_", DiagNormal(mu, sigma))
    p_b = pyro.sample("bias_", DiagNormal(bias_mu, bias_sigma))
    # exp to get log bernoulli
    reg = torch.addmm(p_b, x_data, p_w)
    latent = sigmoid(reg)
    pyro.observe("categorical", Categorical(latent), y_data.unsqueeze(1))


def guide(data):
    #sample from approximate posterior for weights
    x_data = data[:,:-1]
    w_mu = Variable(torch.randn(D, 1), requires_grad=True)
    w_sig = Variable(-3.0*torch.ones(D,1) + 0.05*torch.randn(D, 1), requires_grad=True)
    b_mu = Variable(torch.randn(1), requires_grad=True)
    b_sig = Variable(-3.0*torch.ones(1) + 0.05 * torch.randn(1), requires_grad=True)
    mw_param = pyro.param("guide_mean_weight", w_mu)
    sw_param = pyro.param("guide_sigma_weight", w_sig)
    mb_param  = pyro.param("guide_mean_bias", b_mu)
    sb_param = pyro.param("guide_sigma__bias", b_sig)
    sw_param = torch.exp(sw_param)
    sb_param = torch.exp(sb_param)
    q_w = pyro.sample("weight_", DiagNormal(mw_param, sw_param))
    q_b = pyro.sample("bias_", DiagNormal(mb_param, sb_param))

# adam_params = {"lr": 0.00001, "betas": (0.95, 0.999)}
adam_params = {"lr": 0.001}
adam_optim = pyro.optim(torch.optim.Adam, adam_params)
sgd_optim = pyro.optim(torch.optim.SGD, adam_params)

# dat = build_toy_dataset(N)
x = df.as_matrix(columns=range(D))
y = df.as_matrix(columns=[D])
raw_data = Variable(torch.Tensor(df.as_matrix().tolist()))
data = normalize(raw_data, 2, dim=1)
x_norm = normalize(Variable(torch.Tensor(x.tolist())), 2, dim=1)
y = Variable(torch.Tensor(y.tolist()))
data = torch.cat((x_norm, y), 1)

def inspect_post_params(data):
    mw_param = pyro.param("guide_mean_weight")
    sw_param = pyro.param("guide_sigma_weight")
    mb_param  = pyro.param("guide_mean_bias")
    sb_param = pyro.param("guide_sigma__bias")
    tuples = [("weight mean", mw_param), ("weight sigma", sw_param),\
              ("mean bias", mb_param), ("bias sigma", sb_param)]
    return iter(tuples)

nr_samples = 50
nr_epochs = 50
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
    print("epoch avg loss {}".format(epoch_loss/float(N)))
