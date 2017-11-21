from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.autograd import Variable
import numpy as np

import pyro.distributions as dist
from pyro.contrib.scoped import SVI
from pyro.optim import Adam
from pyro.contrib import named

# Implements HMM with a naive bayes observation model using a mean-field variational factorization.

K = 6
K_vs = np.arange(K)
M = 3
M_vs = np.arange(M)

# Make up some parameters.

# Start in state 0
alpha_pi = [100, 1, 1, 1, 1, 1]

# Favor transitioning to next state.
alpha_trans = []
for k in range(K):
    trans = [0.1 for _ in range(K)]
    trans[(k + 1) % K] = 100
    alpha_trans.append(trans)

# Favor emitting same values.
alpha_obs = []
for k in range(K):
    obs = [0.1 for _ in range(M)]
    obs[k % M] = 100
    alpha_obs.append(obs)


def model(latent, data):
    # Parameters.
    p = latent.parameters
    p.pi.sample_(dist.dirichlet, Variable(torch.Tensor(alpha_pi)))
    p.tau = named.List(size=K)
    p.omega = named.List(size=K)
    for k in range(K):
        p.tau[k].sample_(dist.dirichlet, Variable(torch.Tensor(alpha_trans[k])))
        p.omega[k].sample_(dist.dirichlet, Variable(torch.Tensor(alpha_obs[k])))

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
        for j, dat, x in step.obs.ienumerate_(data[t]):
            x.observe_(dist.categorical, [dat], p.omega[z[0]], vs=M_vs)


def guide(latent, data):
    # Organization
    def var(v):
        return Variable(torch.Tensor(v), requires_grad=True)
    v = latent.variational_parameters
    p = latent.parameters

    # Mean field for parameters.
    p.pi.sample_(dist.dirichlet,
                 v.log_alpha_pi.param_(var([-1] * K)).exp())

    v.log_alpha_tau = named.List(size=K)
    v.log_alpha_omega = named.List(size=K)
    p.tau = named.List(size=K)
    p.omega = named.List(size=K)
    for k in range(K):
        p.tau[k].sample_(dist.dirichlet,
                         v.log_alpha_tau[k].param_(var([-1] * K)).exp())

        p.omega[k].sample_(dist.dirichlet,
                           v.log_alpha_omega[k].param_(var([-1] * M)).exp())

    # Mean field for hidden states.
    T = len(data)
    latent.step = named.List(size=T)
    v.step = named.List(size=T)
    for t in range(len(data)):
        scores = v.step[t].log_z.param_(var([-1] * K)).exp()
        latent.step[t].z.sample_(dist.categorical, scores.div(scores.sum()), vs=K_vs)


def main(args):
    optim = Adam({"lr": 0.01})

    inference = SVI(model, guide, optim, loss="ELBO")
    data = [[0, 0, 0], [1, 1, 1], [2, 2, 2]] * 10

    # Sanity check.
    # temp = named.Object("temp")
    # print(data)
    # model(temp, data)
    # print(repr(temp))

    print('Step\tLoss')
    total = 0
    for step in range(args.num_epochs):
        loss = inference.step(data)
        total += loss
        if step % 100 == 0:
            print('{}\t{:0.5g}'.format(step, total / 100.0))
            total = 0

            # Print the parameters each time.
            print(inference._guide_latent[0].variational_parameters.log_alpha_pi)
            for i, step in enumerate(inference._guide_latent[0].step):
                print(i, ":", step.z, end=" ")
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    args = parser.parse_args()
    main(args)
