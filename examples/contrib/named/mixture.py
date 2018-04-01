from __future__ import absolute_import, division, print_function

import argparse

import torch

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
    latent.probs.param_(torch.tensor(torch.ones(k) / k, requires_grad=True))
    latent.locs.param_(torch.zeros(k, requires_grad=True))
    latent.scales.param_(torch.ones(k, requires_grad=True))

    # Observe all the data. We pass a local latent in to the local_model.
    latent.local = named.List()
    for x in data:
        local_model(latent.local.add(), latent.probs, latent.locs, latent.scales, obs=x)


def local_model(latent, ps, locs, scales, obs=None):
    i = latent.id.sample_(dist.Categorical(softmax(ps)))
    return latent.x.sample_(dist.Normal(locs[i], scales[i]), obs=obs)


def guide(data, k):
    latent = named.Object("latent")
    latent.local = named.List()
    for x in data:
        # We pass a local latent in to the local_guide.
        local_guide(latent.local.add(), k)


def local_guide(latent, k):
    # The local guide simply guesses category assignments.
    latent.probs.param_(torch.tensor(torch.ones(k) / k, requires_grad=True))
    latent.id.sample_(dist.Categorical(softmax(latent.probs)))


def main(args):
    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss="ELBO")
    data = torch.tensor([0.0, 1.0, 2.0, 20.0, 30.0, 40.0])
    k = 2

    print('Step\tLoss')
    for step in range(args.num_epochs):
        if step % 1000 == 0:
            loss = inference.step(data, k)
            print('{}\t{:0.5g}'.format(step, loss))

    print('Parameters:')
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        print('{} = {}'.format(name, pyro.param(name).detach().cpu().numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10000, type=int)
    args = parser.parse_args()
    main(args)
