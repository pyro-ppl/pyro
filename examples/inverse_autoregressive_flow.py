from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import pyro
from pyro.distributions import DiagNormal
from pyro.infer.kl_qp import KL_QP
import torch.optim as optim
from pyro.distributions.transformed_distribution import InverseAutoregressiveFlow
from pyro.distributions.transformed_distribution import TransformedDistribution
import torch.nn as nn
from pyro.util import ng_ones, ng_zeros
import sys
import visdom
from sklearn.datasets import make_moons

visualize = False

if visualize:
    viz = visdom.Visdom()

    def make_scatter_plot(X, title):
        win = viz.scatter(X=X,
                          opts=dict(markersize=2, title=title))

dim_x = 2
dim_z = 10
dim_h_decoder = 50
dim_h_encoder = 50
dim_h_iaf = 10

def gen_distr(N):
    x, y = make_moons(n_samples=2 * N, noise=0.2)
    data = x[y == 0]
    return torch.Tensor(data)

N = 500
batch_size = 5
data = Variable(gen_distr(N))
#mini_batches = [data[i*batch_size:(i+1)*batch_size] for i in range(N/batch_size)]
n_mini_batches = N / batch_size

use_iaf = True
if use_iaf:
    pt_iaf = InverseAutoregressiveFlow(dim_z, dim_h_iaf)

#make_scatter_plot(data.data.numpy(), "training data")

pt_decoder = nn.Sequential(nn.Linear(dim_z, dim_h_decoder), nn.Softplus(),
                           #nn.Linear(dim_h_decoder, dim_h_decoder), nn.Softplus(),
                           nn.Linear(dim_h_decoder, 2 * dim_x))
pt_encoder = nn.Sequential(nn.Linear(dim_x, dim_h_encoder), nn.Softplus(),
                           #nn.Linear(dim_h_encoder, dim_h_encoder), nn.Softplus(),
                           nn.Linear(dim_h_encoder, 2 * dim_z))

def model(observed_data):
    decoder = pyro.module("decoder", pt_decoder)

    z_prior = DiagNormal(ng_zeros(observed_data.size(0), dim_z),
                         ng_ones(observed_data.size(0), dim_z))
    z = pyro.sample("z", z_prior)
    z_decoded = decoder(z)
    mu_x = z_decoded[:, 0:dim_x]
    sigma_x = torch.exp(z_decoded[:, dim_x:])
    obs_dist = DiagNormal(mu_x, sigma_x)
    pyro.map_data("map", observed_data, lambda i, x: pyro.observe("obs", obs_dist, x))

def guide(observed_data):
    encoder = pyro.module("encoder", pt_encoder)

    x_encoded = encoder(observed_data)
    mu_z = x_encoded[:, 0:dim_z]
    sigma_z = torch.exp(x_encoded[:, dim_z:])
    if use_iaf:
        iaf = pyro.module("iaf", pt_iaf)
        z_dist = TransformedDistribution(DiagNormal(mu_z, sigma_z), iaf)
    else:
        z_dist = DiagNormal(mu_z, sigma_z)
    z = pyro.sample("z", z_dist)
    pyro.map_data("map", observed_data, lambda i, x: None)

def sample_x(n_samples, return_mu=False):
    z_prior = DiagNormal(ng_zeros(n_samples, dim_z),
                         ng_ones(n_samples, dim_z))
    z = z_prior.sample()
    decoder = pyro.module("decoder", pt_decoder)
    z_decoded = decoder(z)
    mu_x = z_decoded[:, 0:dim_x]
    sigma_x = torch.exp(z_decoded[:, dim_x:])
    obs_dist = DiagNormal(mu_x, sigma_x)
    if not return_mu:
        x = obs_dist.sample()
        return x
    else:
        return mu_x

n_steps = 1000
kl_optim = KL_QP(model, guide, pyro.optim(optim.Adam, {"lr": 0.01, "betas": (0.90, 0.999)}))
for step in range(n_steps):
    losses = []
    for mini_batch in range(n_mini_batches):
        loss = kl_optim.step(data)
        losses.append(loss)
    if step % 10 == 0:
        print("[epoch %04d] elbo = %.4f" % (step, -np.mean(losses)))
        sys.stdout.flush()

if visualize:
    samples = sample_x(N)
    samples_mu = sample_x(N, return_mu=True)
    title = "x samples "
    addendum = "(with flow)" if use_iaf else "(without flow)"
    make_scatter_plot(samples.data.numpy(), title + addendum)
    make_scatter_plot(samples_mu.data.numpy(), title + addendum + '[mu]')
