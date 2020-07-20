# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of `NeuTraReparam` to run neural transport HMC [1]
on a toy model that draws from a banana-shaped bivariate distribution [2]. We first
train an autoguide by using `AutoNormalizingFlow` that learns a transformation from
a simple latent space (isotropic gaussian) to the more complex geometry of the
posterior. Subsequently, we use `NeuTraReparam` to run HMC and draw samples from this
simplified "warped" posterior. Finally, we use our learnt transformation to transform
these samples back to the original space. For comparison, we also draw samples from
a NeuTra-reparametrized model that uses a much simpler `AutoDiagonalNormal` guide.

References:
----------
[1] Hoffman, M., Sountsov, P., Dillon, J. V., Langmore, I., Tran, D., and Vasudevan,
 S. Neutra-lizing bad geometry in hamiltonian monte carlo using neural transport.
 arXiv preprint arXiv:1903.03704, 2019.
[2] Wang Z., Broccardo M., and Song J. Hamiltonian Monte Carlo Methods for Subset
 Simulation in Reliability Analysis. arXiv preprint arXiv:1706.01435, 2018.
"""

import argparse
import logging
import os
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec

import pyro
import pyro.distributions as dist
from pyro import optim, poutine
from pyro.distributions import constraints
from pyro.distributions.transforms import block_autoregressive, iterated
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormalizingFlow
from pyro.infer.reparam import NeuTraReparam

logging.basicConfig(format='%(message)s', level=logging.INFO)


class BananaShaped(dist.TorchDistribution):
    support = constraints.real_vector

    def __init__(self, a, b, rho=0.9):
        self.a = a
        self.b = b
        self.rho = rho
        self.mvn = dist.MultivariateNormal(torch.tensor([0., 0.]),
                                           covariance_matrix=torch.tensor([[1., self.rho], [self.rho, 1.]]))
        super().__init__(event_shape=(2,))

    def sample(self, sample_shape=()):
        u = self.mvn.sample(sample_shape)
        u0, u1 = u[..., 0], u[..., 1]
        a, b = self.a, self.b
        x = a * u0
        y = (u1 / a) + b * (u0 ** 2 + a ** 2)
        return torch.stack([x, y], -1)

    def log_prob(self, x):
        x, y = x[..., 0], x[..., 1]
        a, b = self.a, self.b
        u0 = x / a
        u1 = (y - b * (u0 ** 2 + a ** 2)) * a
        return self.mvn.log_prob(torch.stack([u0, u1], dim=-1))


def model(a, b, rho=0.9):
    pyro.sample('x', BananaShaped(a, b, rho))


def fit_guide(guide, args):
    pyro.clear_param_store()
    adam = optim.Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, adam, Trace_ELBO())
    for i in range(args.num_steps):
        loss = svi.step(args.param_a, args.param_b)
        if i % 500 == 0:
            logging.info("[{}]Elbo loss = {:.2f}".format(i, loss))


def run_hmc(args, model):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, warmup_steps=args.num_warmup, num_samples=args.num_samples)
    mcmc.run(args.param_a, args.param_b)
    mcmc.summary()
    return mcmc


def main(args):
    pyro.set_rng_seed(args.rng_seed)
    fig = plt.figure(figsize=(8, 16), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, 0])
    ax8 = fig.add_subplot(gs[3, 1])
    xlim = tuple(int(x) for x in args.x_lim.strip().split(','))
    ylim = tuple(int(x) for x in args.y_lim.strip().split(','))
    assert len(xlim) == 2
    assert len(ylim) == 2

    # 1. Plot samples drawn from BananaShaped distribution
    x1, x2 = torch.meshgrid([torch.linspace(*xlim, 100), torch.linspace(*ylim, 100)])
    d = BananaShaped(args.param_a, args.param_b)
    p = torch.exp(d.log_prob(torch.stack([x1, x2], dim=-1)))
    ax1.contourf(x1, x2, p, cmap='OrRd',)
    ax1.set(xlabel='x0', ylabel='x1', xlim=xlim, ylim=ylim,
            title='BananaShaped distribution: \nlog density')

    # 2. Run vanilla HMC
    logging.info('\nDrawing samples using vanilla HMC ...')
    mcmc = run_hmc(args, model)
    vanilla_samples = mcmc.get_samples()['x'].cpu().numpy()
    ax2.contourf(x1, x2, p, cmap='OrRd')
    ax2.set(xlabel='x0', ylabel='x1', xlim=xlim, ylim=ylim,
            title='Posterior \n(vanilla HMC)')
    sns.kdeplot(vanilla_samples[:, 0], vanilla_samples[:, 1], ax=ax2)

    # 3(a). Fit a diagonal normal autoguide
    logging.info('\nFitting a DiagNormal autoguide ...')
    guide = AutoDiagonalNormal(model, init_scale=0.05)
    fit_guide(guide, args)
    with pyro.plate('N', args.num_samples):
        guide_samples = guide()['x'].detach().cpu().numpy()

    ax3.contourf(x1, x2, p, cmap='OrRd')
    ax3.set(xlabel='x0', ylabel='x1', xlim=xlim, ylim=ylim,
            title='Posterior \n(DiagNormal autoguide)')
    sns.kdeplot(guide_samples[:, 0], guide_samples[:, 1], ax=ax3)

    # 3(b). Draw samples using NeuTra HMC
    logging.info('\nDrawing samples using DiagNormal autoguide + NeuTra HMC ...')
    neutra = NeuTraReparam(guide.requires_grad_(False))
    neutra_model = poutine.reparam(model, config=lambda _: neutra)
    mcmc = run_hmc(args, neutra_model)
    zs = mcmc.get_samples()['x_shared_latent']
    sns.scatterplot(zs[:, 0], zs[:, 1], alpha=0.2, ax=ax4)
    ax4.set(xlabel='x0', ylabel='x1',
            title='Posterior (warped) samples \n(DiagNormal + NeuTra HMC)')

    samples = neutra.transform_sample(zs)
    samples = samples['x'].cpu().numpy()
    ax5.contourf(x1, x2, p, cmap='OrRd')
    ax5.set(xlabel='x0', ylabel='x1', xlim=xlim, ylim=ylim,
            title='Posterior (transformed) \n(DiagNormal + NeuTra HMC)')
    sns.kdeplot(samples[:, 0], samples[:, 1], ax=ax5)

    # 4(a). Fit a BNAF autoguide
    logging.info('\nFitting a BNAF autoguide ...')
    guide = AutoNormalizingFlow(model, partial(iterated, args.num_flows, block_autoregressive))
    fit_guide(guide, args)
    with pyro.plate('N', args.num_samples):
        guide_samples = guide()['x'].detach().cpu().numpy()

    ax6.contourf(x1, x2, p, cmap='OrRd')
    ax6.set(xlabel='x0', ylabel='x1', xlim=xlim, ylim=ylim,
            title='Posterior \n(BNAF autoguide)')
    sns.kdeplot(guide_samples[:, 0], guide_samples[:, 1], ax=ax6)

    # 4(b). Draw samples using NeuTra HMC
    logging.info('\nDrawing samples using BNAF autoguide + NeuTra HMC ...')
    neutra = NeuTraReparam(guide.requires_grad_(False))
    neutra_model = poutine.reparam(model, config=lambda _: neutra)
    mcmc = run_hmc(args, neutra_model)
    zs = mcmc.get_samples()['x_shared_latent']
    sns.scatterplot(zs[:, 0], zs[:, 1], alpha=0.2, ax=ax7)
    ax7.set(xlabel='x0', ylabel='x1', title='Posterior (warped) samples \n(BNAF + NeuTra HMC)')

    samples = neutra.transform_sample(zs)
    samples = samples['x'].cpu().numpy()
    ax8.contourf(x1, x2, p, cmap='OrRd')
    ax8.set(xlabel='x0', ylabel='x1', xlim=xlim, ylim=ylim,
            title='Posterior (transformed) \n(BNAF + NeuTra HMC)')
    sns.kdeplot(samples[:, 0], samples[:, 1], ax=ax8)

    plt.savefig(os.path.join(os.path.dirname(__file__), 'neutra.pdf'))


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description='Example illustrating NeuTra Reparametrizer')
    parser.add_argument('-n', '--num-steps', default=10000, type=int,
                        help='number of SVI steps')
    parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float,
                        help='learning rate for the Adam optimizer')
    parser.add_argument('--rng-seed', default=1, type=int,
                        help='RNG seed')
    parser.add_argument('--num-warmup', default=500, type=int,
                        help='number of warmup steps for NUTS')
    parser.add_argument('--num-samples', default=1000, type=int,
                        help='number of samples to be drawn from NUTS')
    parser.add_argument('--param-a', default=1.15, type=float,
                        help='parameter `a` of BananaShaped distribution')
    parser.add_argument('--param-b', default=1., type=float,
                        help='parameter `b` of BananaShaped distribution')
    parser.add_argument('--num-flows', default=1, type=int,
                        help='number of flows in the BNAF autoguide')
    parser.add_argument('--x-lim', default='-3,3', type=str,
                        help='x limits for the plots')
    parser.add_argument('--y-lim', default='0,8', type=str,
                        help='y limits for the plots')

    args = parser.parse_args()
    main(args)
