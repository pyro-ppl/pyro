from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.contrib import named
from pyro.distributions.util import softmax
from pyro.infer import SVI
from pyro.optim import Adam

# This is a simple gaussian mixture model.
#
# The example demonstrates how to pass named.Objects() from a global model to
# a local model implemented as a helper function.


def model(data, k):
    latent = named.Object("latent")

    # Create parameters for a Gaussian mixture model.
    latent.ps.param_(Variable(torch.ones(k) / k, requires_grad=True))
    latent.mus.param_(Variable(torch.zeros(k), requires_grad=True))
    latent.sigmas.param_(Variable(torch.ones(k), requires_grad=True))

    # Observe all the data. We pass a local latent in to the local_model.
    latent.local = named.List()
    for x in data:
        local_model(latent.local.add(), latent.ps, latent.mus, latent.sigmas, obs=x)


def local_model(latent, ps, mus, sigmas, obs=None):
    i = latent.id.sample_(dist.Categorical(softmax(ps)))
    return latent.x.sample_(dist.Normal(mus[i], sigmas[i]), obs=obs)


def guide(data, k):
    latent = named.Object("latent")
    latent.local = named.List()
    for x in data:
        # We pass a local latent in to the local_guide.
        local_guide(latent.local.add(), k)


def local_guide(latent, k):
    # The local guide simply guesses category assignments.
    latent.ps.param_(Variable(torch.ones(k) / k, requires_grad=True))
    latent.id.sample_(dist.Categorical(softmax(latent.ps)))


def main(args):
    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss="ELBO")
    data = Variable(torch.Tensor([0, 1, 2, 20, 30, 40]))
    k = 2

    print('Step\tLoss')
    for step in range(args.num_epochs):
        if step % 1000 == 0:
            loss = inference.step(data, k)
            print('{}\t{:0.5g}'.format(step, loss))

    print('Parameters:')
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        print('{} = {}'.format(name, pyro.param(name).data.cpu().numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10000, type=int)
    args = parser.parse_args()
    main(args)
