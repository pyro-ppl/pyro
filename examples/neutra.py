# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to use a trained AutoIAFNormal autoguide to transform a posterior to a
Gaussian-like one. The transform will be used to get better mixing rate for NUTS sampler.

**References:**

    1. Hoffman, M. et al. (2019), "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport",
       (https://arxiv.org/abs/1903.03704)
"""


import argparse
import logging

import numpy as np
import torch
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

import pyro
from pyro import optim, poutine
from pyro.infer.autoguide import AutoIAFNormal
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.reparam import NeuTraReparam

logging.basicConfig(format='%(message)s',  level=logging.INFO)


class DualMoonDistribution(dist.TorchDistribution):
    support = constraints.real_vector

    def __init__(self):
        super(DualMoonDistribution, self).__init__(event_shape=(2,))

    def sample(self, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        return torch.zeros(sample_shape + self.event_shape)

    def log_prob(self, x):
        term1 = 0.5 * ((torch.norm(x, dim=-1) - 2) / 0.4) ** 2
        term2 = -0.5 * ((x[..., :1] + torch.tensor([-2., 2.])) / 0.6) ** 2
        pe = term1 - torch.logsumexp(term2, dim=-1)
        return -pe


def dual_moon_model():
    pyro.sample('x', DualMoonDistribution())


def main(args):
    logging.info("Sampling from vanilla HMC...")
    nuts_kernel = NUTS(dual_moon_model)
    mcmc = MCMC(nuts_kernel, args.num_warmup, args.num_samples)
    mcmc.run()
    mcmc.summary()
    vanilla_samples = mcmc.get_samples()['x'].detach().numpy()

    adam = optim.Adam({'lr': 1e-3})
    guide = AutoIAFNormal(dual_moon_model, num_flows=args.num_flows, hidden_dim=args.num_hidden)
    svi = SVI(dual_moon_model, guide, adam, Trace_ELBO())
    logging.info("Training  AutoIAFNormal guide...")
    for i in range(args.num_iters):
        loss = svi.step()
        if i % 500 == 0:
            logging.info("[{}]Elbo loss = {:.2f}".format(i, loss))
    # Get samples from the trained guide for plotting
    with pyro.plate('N', args.num_samples):
        guide_samples = guide()['x'].detach().numpy()

    logging.info("Sampling from NeuTra HMC...")
    neutra = NeuTraReparam(guide)
    model = poutine.reparam(dual_moon_model, config=lambda _: neutra)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, args.num_warmup, args.num_samples)
    mcmc.run()
    mcmc.summary()
    zs = mcmc.get_samples()["x_shared_latent"]
    logging.info("Transform samples into unwarped space...")
    samples = neutra.transform_sample(zs)
    zs = zs.detach().numpy()
    samples = samples['x'].detach().numpy()

    # make plots

    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    P = torch.exp(DualMoonDistribution().log_prob(torch.from_numpy(np.stack([X1, X2], axis=-1))))

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.contourf(X1, X2, P, cmap='OrRd')
    sns.kdeplot(guide_samples[:, 0], guide_samples[:, 1], n_levels=30, ax=ax1)
    ax1.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='Posterior using AutoIAFNormal guide')

    ax2.contourf(X1, X2, P, cmap='OrRd')
    sns.kdeplot(vanilla_samples[:, 0], vanilla_samples[:, 1], n_levels=30, ax=ax2)
    ax2.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='Posterior using vanilla HMC sampler')

    sns.scatterplot(zs[:, 0], zs[:, 1], ax=ax3, hue=samples[:, 0] < 0.,
                    s=30, alpha=0.5, edgecolor="none")
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, ['left moon', 'right moon'])
    ax3.set(xlim=[-8, 8], ylim=[-4, 4],
            xlabel='x0', ylabel='x1', title='Samples from the warped posterior - p(z)')

    ax4.contourf(X1, X2, P, cmap='OrRd')
    sns.kdeplot(samples[:, 0], samples[:, 1], n_levels=30, ax=ax4)
    ax4.set(xlim=[-3, 3], ylim=[-3, 3],
            xlabel='x0', ylabel='x1', title='Posterior using NeuTra HMC sampler')

    plt.savefig("neutra.pdf")
    plt.close()


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="NeuTra HMC")
    parser.add_argument('-n', '--num-samples', nargs='?', default=3000, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=1000, type=int)
    parser.add_argument('--num-flows', nargs='?', default=2, type=int)
    parser.add_argument('--num-hidden', nargs='?', default=10, type=int)
    parser.add_argument('--num-iters', nargs='?', default=6000, type=int)
    args = parser.parse_args()

    main(args)
