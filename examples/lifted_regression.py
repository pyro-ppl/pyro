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

N = 500 # data

data = build_toy_dataset(N)

class LogisticRegression(nn.Module) :
    def __init__(self, dim) :
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data) :
        return self.sigmoid(self.linear(data))

regression = LogisticRegression(1)

# TODO: use larger sigma
# TODO: use randn/softplus insetad of 0/1?
# def prior(tensor) :
#     mu = Variable(torch.zeros(tensor.size()))
#     sigma = Variable(torch.ones(tensor.size()))
#     return DiagNormal(mu, sigma)()

class prior(pyro.distributions.Distribution):
    def __init__(self):
        super(prior, self).__init__()
#
    def sample(self, tensor):
        m = Variable(torch.zeros(tensor.size()))
        s = Variable(torch.ones(tensor.size()))
        # dont have this info during initialization but needed to score
        self.dist = DiagNormal(m, s)
        return DiagNormal(m, s)()
#
    def log_pdf(self, x, *args, **kwargs):
        return self.dist.log_pdf(x)

def guide(data) :
    nn_dist = pyro.random_module('name', regression, prior())
    nn = nn_dist()

def model(data):
    """
    Logistic regression model
    """
    x = data[0]
    y = data[1]
    nn_dist = pyro.random_module('name', regression, prior())
    sampled_nn = nn_dist()
    out = sampled_nn(x)

    pyro.observe("bernoulli", Bernoulli(out), y)

def posterior():
    param_dict = pyro.get_param_store()._params
    return param_dict['mu']

adam_params = {"lr": 0.0001}
adam_optim = pyro.optim(torch.optim.Adam, adam_params)
sgd_optim = pyro.optim(torch.optim.SGD, adam_params)

nr_samples = 50
nr_epochs = 500
batch_size = 1

# x = df.as_matrix(columns=range(D))
# y = np.squeeze(df.as_matrix(columns=[D]))
dat = df.as_matrix()

grad_step = KL_QP(model, guide, sgd_optim)

# apply it to minibatches of data by hand:
for j in range(nr_epochs):
    epoch_loss = 0.0
    for i in range(dat[0].size):
        batch_data = Variable(torch.Tensor(dat[0]))
        epoch_loss += grad_step.step(batch_data)
#         print "accum loss",epoch_loss
        bb()
    print("epoch avg loss {}".format(epoch_loss))
