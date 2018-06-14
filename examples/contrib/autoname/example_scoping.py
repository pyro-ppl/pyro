from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from pyro.contrib.autoname import scope


# in this example, we'll see how scoping and non-strict naming
# makes Pyro programs significantly more composable

def model(data, k):
    # Create parameters for a Gaussian mixture model.
    probs = pyro.param("mprobs", torch.ones(k) / k, constraint=constraints.simplex)
    locs = pyro.param("locs", torch.zeros(k))
    scales = pyro.param("scales", torch.ones(k), constraint=constraints.positive)

    # Observe all the data
    for x in data:
        local_model(probs, locs, scales, obs=x)


@scope(prefix="local")
def local_model(ps, locs, scales, obs=None):
    i = pyro.sample("id", dist.Categorical(ps))
    return pyro.sample("x", dist.Normal(locs[i], scales[i]), obs=obs)


def guide(data, k):
    for i, x in enumerate(data):
        # We pass a local param in to the local_guide since scopes only affect sample
        probs = pyro.param("probs_{}".format(i),
                           torch.ones(k) / k, constraint=constraints.positive)
        local_guide(k, probs)


@scope(prefix="local")
def local_guide(k, probs):
    # The local guide simply guesses category assignments.
    return pyro.sample("id", dist.Categorical(probs))


def main(args):
    pyro.set_rng_seed(0)
    pyro.enable_validation()

    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss=Trace_ELBO())
    data = torch.tensor([0.0, 1.0, 2.0, 20.0, 30.0, 40.0])
    k = 2

    print('Step\tLoss')
    loss = 0.0
    for step in range(args.num_epochs):
        if step and step % 10 == 0:
            print('{}\t{:0.5g}'.format(step, loss))
            loss = 0.0
        loss += inference.step(data, k)

    print('Parameters:')
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        print('{} = {}'.format(name, pyro.param(name).detach().cpu().numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=200, type=int)
    args = parser.parse_args()
    main(args)
