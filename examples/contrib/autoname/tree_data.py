from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import named
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# This is a linear mixed-effects model over arbitrary json-like data.
# Data can be a number, a list of data, or a dict with data values.
#
# The goal is to learn a mean field approximation to the posterior
# values z, parameterized by parameters post_loc and post_scale.
#
# Notice that the named.Objects allow for modularity that fits well
# with the recursive model and guide functions.


def model(data):
    latent = named.Object("latent")
    latent.z.sample_(dist.Normal(0.0, 1.0))
    model_recurse(data, latent)


def model_recurse(data, latent):
    if torch.is_tensor(data):
        latent.x.sample_(dist.Normal(latent.z, 1.0), obs=data)
    elif isinstance(data, list):
        latent.prior_scale.param_(torch.tensor(1.0), constraint=constraints.positive)
        latent.list = named.List()
        for data_i in data:
            latent_i = latent.list.add()
            latent_i.z.sample_(dist.Normal(latent.z, latent.prior_scale))
            model_recurse(data_i, latent_i)
    elif isinstance(data, dict):
        latent.prior_scale.param_(torch.tensor(1.0), constraint=constraints.positive)
        latent.dict = named.Dict()
        for key, value in data.items():
            latent.dict[key].z.sample_(dist.Normal(latent.z, latent.prior_scale))
            model_recurse(value, latent.dict[key])
    else:
        raise TypeError("Unsupported type {}".format(type(data)))


def guide(data):
    guide_recurse(data, named.Object("latent"))


def guide_recurse(data, latent):
    latent.post_loc.param_(torch.tensor(0.0))
    latent.post_scale.param_(torch.tensor(1.0), constraint=constraints.positive)
    latent.z.sample_(dist.Normal(latent.post_loc, latent.post_scale))
    if torch.is_tensor(data):
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
    pyro.set_rng_seed(0)
    pyro.enable_validation()

    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss=Trace_ELBO())

    # Data is an arbitrary json-like structure with tensors at leaves.
    one = torch.tensor(1.0)
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
    loss = 0.0
    for step in range(args.num_epochs):
        loss += inference.step(data)
        if step and step % 10 == 0:
            print('{}\t{:0.5g}'.format(step, loss))
            loss = 0.0

    print('Parameters:')
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        print('{} = {}'.format(name, pyro.param(name).detach().cpu().numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=100, type=int)
    args = parser.parse_args()
    main(args)
