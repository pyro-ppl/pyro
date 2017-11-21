from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.autograd import Variable

import pyro.distributions as dist
from pyro.contrib import scoped
from pyro.distributions.util import softmax
from pyro.optim import Adam

# This is a simple gaussian mixture model.
#
# The example demonstrates how to pass named.Objects() from a global model to
# a local model implemented as a helper function.


def model(latent, data, k):
    # Create parameters for a Gaussian mixture model.
    p = latent.params
    p.ps.param_(Variable(torch.ones(k) / k, requires_grad=True))
    p.mus.param_(Variable(torch.zeros(k), requires_grad=True))
    p.sigmas.param_(Variable(torch.ones(k), requires_grad=True))

    # Observe all the data. We pass a local latent in to the local_model.
    for _, x, local in latent.local.ienumerate_(data):
        i = local.id.sample_(dist.categorical, softmax(p.ps), one_hot=False)
        local.x.sample_(dist.normal, p.mus[i], p.sigmas[i], obs=x)


def guide(latent, data, k):
    for _, _, local in latent.local.ienumerate_(data):
        local.ps.param_(Variable(torch.ones(k) / k, requires_grad=True))
        local.id.sample_(dist.categorical, softmax(local.ps), one_hot=False)


def main(args):
    optim = Adam({"lr": 0.1})
    inference = scoped.SVI(model, guide, optim, loss="ELBO")
    data = Variable(torch.Tensor([0, 1, 2, 20, 30, 40]))
    k = 2

    print('Step\tLoss')
    for step in range(args.num_epochs):
        if step % 1000 == 0:
            loss = inference.step(data, k)
            print('{}\t{:0.5g}'.format(step, loss))

    print('Parameters:')
    print(repr(inference._model_latent[0].params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10000, type=int)
    args = parser.parse_args()
    main(args)
