from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.autograd import Variable
import numpy as np

import pyro
import pyro.distributions as dist
from pyro.contrib.scoped import SVI
from pyro.optim import Adam
from pyro.contrib import named

# Implements HMM with a naive bayes observation model using a mean-field variational factorization.

K = 6
K_vs = np.arange(K)
M = 3
M_vs = np.arange(M)


def model(latent, data):
    # Parameters.
    p = latent.parameters
    p.pi.sample_(dist.dirichlet, Variable(torch.Tensor([10, 1, 1, 1, 1, 1])))
    p.tau = named.List()
    p.omega = named.List()
    for k in range(K):
        # Make parameters loopy (learn in future).
        trans = [0.1] * K
        trans[(k + 1) % K] += 10
        p.tau.add().sample_(dist.dirichlet, Variable(torch.Tensor(trans)))
        m = [0.1] * M
        m[k % 3] += 10
        p.omega.add().sample_(dist.dirichlet, Variable(torch.Tensor(m)))

    # Model
    latent.step = named.List()
    z = None
    for t in range(len(data)):
        step = latent.step.add()

        # Transition
        z = step.z.sample_(dist.categorical,
                           p.pi if z is None else p.tau[z[0]],
                           vs=K_vs)

        # Observation
        step.obs = named.List()
        for j, x in step.obs.irange_(len(data[t])):
            x.observe_(dist.categorical, [data[t][j]], p.omega[z[0]],
                       vs=M_vs)


def guide(latent, data):
    # Organization
    def var(v):
        return Variable(torch.Tensor(v), requires_grad=True)
    v = latent.variational_parameters
    p = latent.parameters

    # Mean field for parameters
    p.pi.sample_(dist.dirichlet,
                 v.alpha_pi.param_(var([-1] * K)).exp())

    v.alpha_tau = named.List()
    p.tau = named.List()
    v.alpha_omega = named.List()
    p.omega = named.List()

    for k in range(K):
        p.tau.add().sample_(dist.dirichlet,
                            v.alpha_tau.add().param_(var([-1] * K)).exp())

        p.omega.add().sample_(dist.dirichlet,
                              v.alpha_omega.add().param_(var([-1] * M)).exp())

    # Mean field for hidden states.
    latent.step = named.List()
    v.step = named.List()
    for t in range(len(data)):
        step = latent.step.add()
        vstep = v.step.add()
        scores = vstep.z.param_(var([-1] * K)).exp()
        probs = scores.div(scores.sum())
        step.z.sample_(dist.categorical, probs,
                       vs=K_vs)


def main(args):
    optim = Adam({"lr": 0.001})
    inference = SVI(model, guide, optim, loss="ELBO")
    data = [[0], [1], [2]] * 20

    print('Step\tLoss')
    total = 0
    for step in range(args.num_epochs):
        loss = inference.step(data)
        total += loss
        if step % 100 == 0:
            print('{}\t{:0.5g}'.format(step, total / 100.0))
            total = 0

    print('Parameters:')
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        print('{} = {}'.format(name, pyro.param(name).data.cpu().numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    args = parser.parse_args()
    main(args)
