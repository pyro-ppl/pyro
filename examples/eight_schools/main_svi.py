from __future__ import absolute_import, division, print_function

import argparse
import logging
import sys

import torch

import pyro
import pyro.distributions as dist
from data import J, sigma_tensor, y_tensor
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

softplus = torch.nn.Softplus()
data = torch.stack([y_tensor, sigma_tensor], dim=1)


def model(data):
    y = data[:, 0]
    sigma = data[:, 1]

    eta_dist = dist.Normal(torch.zeros(J), torch.ones(J))
    mu_dist = dist.Normal(loc=torch.zeros(1), scale=10 * torch.ones(1))
    tau_dist = dist.HalfCauchy(loc=torch.zeros(1), scale=25 * torch.ones(1))

    eta = pyro.sample('eta', eta_dist)
    mu = pyro.sample('mu', mu_dist)
    tau = pyro.sample('tau', tau_dist)

    theta = mu + tau * eta

    pyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def guide(data):
    eta_loc = torch.randn(J, 1)
    # note that we initialize our scales to be pretty narrow
    eta_log_sig = torch.tensor(-3.0 * torch.ones(J, 1) + 0.05 * torch.randn(J, 1))
    mu_loc = torch.randn(1)
    mu_log_sig = torch.tensor(-3.0 * torch.ones(1) + 0.05 * torch.randn(1))
    logtau_loc = torch.randn(1)
    logtau_log_sig = torch.tensor(-3.0 * torch.ones(1) + 0.05 * torch.randn(1))

    # register learnable params in the param store
    m_eta_param = pyro.param("guide_mean_eta", eta_loc)
    s_eta_param = softplus(pyro.param("guide_log_scale_eta", eta_log_sig))
    m_mu_param = pyro.param("guide_mean_mu", mu_loc)
    s_mu_param = softplus(pyro.param("guide_log_scale_mu", mu_log_sig))
    m_logtau_param = pyro.param("guide_mean_logtau", logtau_loc)
    s_logtau_param = softplus(pyro.param("guide_log_scale_logtau", logtau_log_sig))

    # guide distributions
    eta_dist = dist.Normal(m_eta_param, s_eta_param)
    mu_dist = dist.Normal(m_mu_param, s_mu_param)
    tau_dist = dist.TransformedDistribution(dist.Normal(m_logtau_param, s_logtau_param),
                                            transforms=torch.distributions.transforms.ExpTransform())

    pyro.sample('eta', eta_dist)
    pyro.sample('mu', mu_dist)
    pyro.sample('tau', tau_dist)


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Eight Schools MCMC')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--num-epochs', type=int, default=1000, metavar='NUMEPOCHS',
                    help='number of epochs (default: 1000)')
args = parser.parse_args()

optim = Adam({'lr': args.lr})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

pyro.clear_param_store()
for j in range(args.num_epochs):
    loss = svi.step(data)
    if j % 100 == 0:
        print("[epoch %04d] loss: %.4f" % (j + 1, loss))

for name in pyro.get_param_store().get_all_param_names():
    print(name)
    print(pyro.param(name).data.numpy())
