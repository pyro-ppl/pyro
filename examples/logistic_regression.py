from pdb import set_trace as bb
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import normalize

from torch.autograd import Variable
import pandas as pd

import pyro
from pyro import poutine
from pyro.distributions import DiagNormal, Gamma, Bernoulli
from pyro.infer.kl_qp import KL_QP
import torchvision.datasets as dset

"""
Estimates w s.t. y = Xw, using Bayesian regularisation.
Replicates the result in https://arxiv.org/pdf/1310.5438.pdf
"""

# use covtype dataset
fname = "data/covtype/covtype.data"
with open(fname, "r+") as f:
    content = f.read()
#     f.seek(0, 0)
#     f.write(first_line.rstrip('\r\n') + '\n' + content)
df = pd.read_csv(fname, header=None)
# def load_ds():
#     for i,row in df.iterrows():
#         yield Variable(torch.Tensor(row[0]))

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
    bias_mu = Variable(torch.zeros(batch_size, 1))
    bias_sigma = Variable(10.0*torch.ones(batch_size, 1))
#     mw_param = pyro.param("mean_weight", mu)
#     sw_param = pyro.param("sigma_weight", sigma)
#     mb_param = pyro.param("mean_bias", bias_mu)
#     sb_param = pyro.param("sigma_bias", bias_sigma)
    p_w = pyro.sample("weight_", DiagNormal(mu, sigma))
    p_b = pyro.sample("bias_", DiagNormal(bias_mu, bias_sigma))
    # exp to get log bernoulli
#     data = data.squeeze()
#     p_w = p_w.squeeze()
#     prob = torch.exp(torch.dot(data, p_w) + p_b)
    reg = torch.addmm(p_b, x_data, p_w)
    latent = sigmoid(reg)
#     prob = torch.exp(torch.addmm(p_b, x_data, p_w))
#     latent = prob / (prob + 1)
#     bb()
    pyro.observe("bernoulli", Bernoulli(latent), y_data.unsqueeze(1))


def guide(data):
    #sample from approximate posterior for weights
    w_mu = Variable(torch.randn(D, 1), requires_grad=True)
    w_sig = Variable(-3.0*torch.ones(D,1) + 0.05*torch.randn(D, 1), requires_grad=True)
    b_mu = Variable(torch.randn(batch_size, 1), requires_grad=True)
    b_sig = Variable(-3.0*torch.ones(batch_size,1) + 0.05 * torch.randn(batch_size, 1), requires_grad=True)
    mw_param = pyro.param("guide_mean_weight", w_mu)
    sw_param = pyro.param("guide_sigma_weight", w_sig)
    mb_param  = pyro.param("guide_mean_bias", b_mu)
    sb_param = pyro.param("guide_sigma__bias", b_sig)
    sw_param = torch.exp(sw_param)
    sb_param = torch.exp(sb_param)
    q_w = pyro.sample("weight_", DiagNormal(mw_param, sw_param))
    q_b = pyro.sample("bias_", DiagNormal(mb_param, sb_param))

# adam_params = {"lr": 0.00001, "betas": (0.95, 0.999)}
adam_params = {"lr": 0.00001}
adam_optim = pyro.optim(torch.optim.Adam, adam_params)
sgd_optim = pyro.optim(torch.optim.SGD, adam_params)

# dat = build_toy_dataset(N)
x = df.as_matrix(columns=range(D))
y = np.squeeze(df.as_matrix(columns=[D]))
raw_data = Variable(torch.Tensor(df.as_matrix()))
data = normalize(raw_data, 2, dim=1)

def posterior(data):
    mu = model(data)
    sample = pyro.sample("sample", Bernoulli(mu))
    return sample

nr_samples = 50
nr_epochs = 600
all_batches = np.arange(0, N, batch_size)
# take care of bad index
if all_batches[-1] != N:
    all_batches = list(all_batches) + [N]

grad_step = KL_QP(model, guide, adam_optim)

# apply it to minibatches of data by hand:
epoch_loss = 0.0
for j in range(nr_epochs):
    for ix, batch_start in enumerate(all_batches[:-1]):
        batch_end = all_batches[ix + 1]
        batch_data = data[batch_start: batch_end]
#         bb()
        epoch_loss += grad_step.step(batch_data)
    print("epoch avg loss {}".format(epoch_loss))
#     bb()

