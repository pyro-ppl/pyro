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
from pyro.util import ng_ones, ng_zeros, ones, zeros
import sys
import visdom
from sklearn.datasets import make_moons

visualize = True

if visualize:
    viz = visdom.Visdom()

    def make_scatter_plot(X, title):
        viz.scatter(X=X,
                    opts=dict(markersize=2, title=title))

dim_x = 2
dim_z = 20
dim_h_iaf = 50


def gen_distr(N):
    x, y = make_moons(n_samples=2 * N, noise=0.2)
    data = x[y == 0]
    return torch.Tensor(data)


N = 250
batch_size = 10
data = Variable(gen_distr(N))
n_mini_batches = N / batch_size
mini_batches = [data[i*batch_size:(i+1)*batch_size] for i in range(n_mini_batches)]
print("n_mini_batches: %d" % n_mini_batches)

N_iafs = 8
pt_iafs = []
for _ in range(N_iafs):
    pt_iaf = InverseAutoregressiveFlow(dim_z, dim_h_iaf)
    pt_iafs.append(pt_iaf)

make_scatter_plot(data.data.numpy(), "training data")

pt_linear = nn.Linear(dim_z, dim_x)

def model(observed_data):
    N_data = observed_data.size(0)
    z_prior = DiagNormal(ng_zeros(N_data, dim_z),
                         ng_ones(N_data, dim_z))
    z = pyro.sample("z", z_prior)
    sigma_x = torch.exp(pyro.param("log_sigma_x", zeros(1, dim_x))).expand(N_data, dim_x)
    linear = pyro.module("linear", pt_linear)
    mu_x = linear(z)
    obs_dist = DiagNormal(mu_x, sigma_x)
    pyro.observe("obs", obs_dist, observed_data)
    #pyro.map_data("map", observed_data, lambda i, x: pyro.observe("obs", obs_dist, x), batch_size=batch_size)


def guide(observed_data):
    N_data = observed_data.size(0)
    mu_z = pyro.param("mu_z", zeros(1, dim_z)).expand(N_data, dim_z)
    sigma_z = torch.exp(pyro.param("log_sigma_z", zeros(1, dim_z))).expand(N_data, dim_z)
    z_dist = DiagNormal(mu_z, sigma_z)
    if N_iafs > 0:
        iafs = [pyro.module("iaf_%d" % i, pt_iaf) for i, pt_iaf in enumerate(pt_iafs)]
        z_dist = TransformedDistribution(z_dist, iafs)
    z = pyro.sample("z", z_dist)
    return z

def sample_x(N_data):
    mu_z = pyro.param("mu_z").expand(N_data, dim_z)
    sigma_z = torch.exp(pyro.param("log_sigma_z")).expand(N_data, dim_z)
    z_dist = DiagNormal(mu_z, sigma_z)
    if N_iafs > 0:
        iafs = [pyro.module("iaf_%d" % i, pt_iaf) for i, pt_iaf in enumerate(pt_iafs)]
        z_dist = TransformedDistribution(z_dist, iafs)
    z = pyro.sample("z", z_dist)
    sigma_x = torch.exp(pyro.param("log_sigma_x"))
    linear = pyro.module("linear", pt_linear)
    mu_x = linear(z)
    obs_dist = DiagNormal(mu_x, sigma_x)
    return obs_dist.sample()


n_steps = 3000
kl_optim = KL_QP(model, guide, pyro.optim(optim.Adam, {"lr": 0.001, "betas": (0.90, 0.999)}), num_particles=1)
for step in range(n_steps):
    losses = []
    for mini_batch in mini_batches:
        loss = kl_optim.step(mini_batch)
        losses.append(loss)
    if step % 10 == 0:
        print("[epoch %04d] elbo = %.4f" % (step, -np.mean(losses)))
        sys.stdout.flush()

    if step % 50 == 0 and visualize:
        samples = sample_x(500)
        make_scatter_plot(samples.data.numpy(), 'x samples')
