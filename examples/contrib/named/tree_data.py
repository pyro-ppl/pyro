from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.contrib import named
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.util import ng_ones, ng_zeros

# This is a linear mixed-effects model over arbitrary json-like data.
# Data can be a number, a list of data, or a dict with data values.
#
# The goal is to learn a mean field approximation to the posterior
# values z, parameterized by parameters post_mu and post_sigma.
#
# Notice that the named.Objects allow for modularity that fits well
# with the recursive model and guide functions.


def model(data):
    latent = named.Object("latent")
    latent.z.sample_(dist.Normal(ng_zeros(1), ng_ones(1)))
    model_recurse(data, latent)


def model_recurse(data, latent):
    if isinstance(data, Variable):
        latent.x.observe_(dist.Normal(latent.z, ng_ones(1)), data)
    elif isinstance(data, list):
        latent.prior_sigma.param_(Variable(torch.ones(1), requires_grad=True))
        latent.list = named.List()
        for data_i in data:
            latent_i = latent.list.add()
            latent_i.z.sample_(dist.Normal(latent.z, latent.prior_sigma))
            model_recurse(data_i, latent_i)
    elif isinstance(data, dict):
        latent.prior_sigma.param_(Variable(torch.ones(1), requires_grad=True))
        latent.dict = named.Dict()
        for key, value in data.items():
            latent.dict[key].z.sample_(dist.Normal(latent.z, latent.prior_sigma))
            model_recurse(value, latent.dict[key])
    else:
        raise TypeError("Unsupported type {}".format(type(data)))


def guide(data):
    guide_recurse(data, named.Object("latent"))


def guide_recurse(data, latent):
    latent.post_mu.param_(Variable(torch.zeros(1), requires_grad=True))
    latent.post_sigma.param_(Variable(torch.ones(1), requires_grad=True))
    latent.z.sample_(dist.Normal(latent.post_mu, latent.post_sigma))
    if isinstance(data, Variable):
        pass
    elif isinstance(data, list):
        latent.list = named.List()
        for datum in data:
            guide_recurse(datum, latent.list.add())
    elif isinstance(data, dict):
        latent.dict = named.Dict()
        for key, value in data.items():
            guide_recurse(value, latent.dict[key])
    else:
        raise TypeError("Unsupported type {}".format(type(data)))


def main(args):
    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss="ELBO")

    # Data is an arbitrary json-like structure with tensors at leaves.
    one = ng_ones(1)
    data = {
        "foo": one,
        "bar": [0 * one, 1 * one, 2 * one],
        "baz": {
            "noun": {
                "concrete": 4 * one,
                "abstract": 6 * one,
            },
            "verb": 2 * one,
        },
    }

    print('Step\tLoss')
    for step in range(args.num_epochs):
        if step % 100 == 0:
            loss = inference.step(data)
            print('{}\t{:0.5g}'.format(step, loss))

    print('Parameters:')
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        print('{} = {}'.format(name, pyro.param(name).data.cpu().numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    args = parser.parse_args()
    main(args)
