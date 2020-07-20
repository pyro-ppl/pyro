# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import torch
from torch.distributions import constraints, transforms

import pyro
import pyro.distributions as dist
from data import J, sigma, y
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

logging.basicConfig(format='%(message)s', level=logging.INFO)
data = torch.stack([y, sigma], dim=1)


def model(data):
    y = data[:, 0]
    sigma = data[:, 1]

    eta = pyro.sample('eta', dist.Normal(torch.zeros(J), torch.ones(J)))
    mu = pyro.sample('mu', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
    tau = pyro.sample('tau', dist.HalfCauchy(scale=25 * torch.ones(1)))

    theta = mu + tau * eta

    pyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def guide(data):
    loc_eta = torch.randn(J, 1)
    # note that we initialize our scales to be pretty narrow
    scale_eta = 0.1 * torch.rand(J, 1)
    loc_mu = torch.randn(1)
    scale_mu = 0.1 * torch.rand(1)
    loc_logtau = torch.randn(1)
    scale_logtau = 0.1 * torch.rand(1)

    # register learnable params in the param store
    m_eta_param = pyro.param("loc_eta", loc_eta)
    s_eta_param = pyro.param("scale_eta", scale_eta, constraint=constraints.positive)
    m_mu_param = pyro.param("loc_mu", loc_mu)
    s_mu_param = pyro.param("scale_mu", scale_mu, constraint=constraints.positive)
    m_logtau_param = pyro.param("loc_logtau", loc_logtau)
    s_logtau_param = pyro.param("scale_logtau", scale_logtau, constraint=constraints.positive)

    # guide distributions
    dist_eta = dist.Normal(m_eta_param, s_eta_param)
    dist_mu = dist.Normal(m_mu_param, s_mu_param)
    dist_tau = dist.TransformedDistribution(dist.Normal(m_logtau_param, s_logtau_param),
                                            transforms=transforms.ExpTransform())

    pyro.sample('eta', dist_eta)
    pyro.sample('mu', dist_mu)
    pyro.sample('tau', dist_tau)


def main(args):
    optim = Adam({'lr': args.lr})
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(model, guide, optim, loss=elbo)

    pyro.clear_param_store()
    for j in range(args.num_epochs):
        loss = svi.step(data)
        if j % 100 == 0:
            logging.info("[epoch %04d] loss: %.4f" % (j + 1, loss))

    for name, value in pyro.get_param_store().items():
        logging.info(name)
        logging.info(value.detach().cpu().numpy())


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description='Eight Schools SVI')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='number of epochs (default: 1000)')
    parser.add_argument('--jit', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
