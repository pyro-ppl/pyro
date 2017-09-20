from pdb import set_trace as bb
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import pandas as pd

import pyro
from pyro import poutine
from pyro.distributions import DiagNormal, Gamma, Bernoulli
from pyro.infer.kl_qp import KL_QP
import torchvision.datasets as dset
import torchvision.transforms as transforms

"""
Estimates w s.t. y = Xw, using Bayesian regularisation.
Replicates the result in https://arxiv.org/pdf/1310.5438.pdf
"""

# use covtype dataset
fname = "data//covtype/covtype.data"
with open(fname, "r+") as f:
    content = f.read()
#     f.seek(0, 0)
#     f.write(first_line.rstrip('\r\n') + '\n' + content)
df = pd.read_csv(fname, header=None)
# df = pd.read_csv("data/covtype/covtype.data")
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
D = 39 # features

def model(i, data):
    """
    Logistic regression model
    """
    mu = Variable(torch.zeros(D))
    sigma = Variable(torch.ones(D)) * 3.0
    p_w = pyro.sample("weight_" + str(i), DiagNormal(mu, sigma))
    p_b = pyro.sample("bias_" + str(i), DiagNormal(mu, sigma))
    # exp to get log bernoulli
    data = data.squeeze()
    p_w = p_w.squeeze()
    prob = torch.exp(torch.dot(data, p_w) + p_b)
    latent = prob / (prob + 1)
    pyro.observe("bernoulli", Bernoulli(latent), data)


def guide(i, data):
    #sample from approximate posterior for weights
    w_mu = Variable(torch.randn(D))
    w_sig = softplus(Variable(torch.randn(D)))
    b_mu = Variable(torch.randn(data.size()) + 10.0)
    b_sig = softplus(Variable(torch.randn(data.size())))
    q_w = pyro.sample("weight_" + str(i), DiagNormal(w_mu, w_sig))
    q_b = pyro.sample("bias_" + str(i), DiagNormal(b_mu, b_sig))

adam_params = {"lr": 0.0001}
adam_optim = pyro.optim(torch.optim.Adam, adam_params)
sgd_optim = pyro.optim(torch.optim.SGD, adam_params)

# dat = build_toy_dataset(N)
x = df.as_matrix(columns=range(D))
y = np.squeeze(df.as_matrix(columns=[D]))
mnist_size = len(x)


nr_samples = 50
nr_epochs = 600
batch_size = 1

# all_batches = np.arange(0, mnist_size, batch_size)

# if all_batches[-1] != mnist_size:
#     all_batches = list(all_batches) + [mnist_size]

grad_step = KL_QP(model, guide, sgd_optim)

# apply it to minibatches of data by hand:
for j in range(nr_epochs):

    epoch_loss = 0.
    for ix in range(dat[0].size(0)):
#         batch_end = all_batches[ix + 1]
         # get batch
#         batch_data = mnist_data[batch_start:batch_end]
        batch_data = dat[0]

        epoch_loss += grad_step.step(ix, batch_data)

    print("epoch avg loss {}".format(epoch_loss / float(mnist_size)))

