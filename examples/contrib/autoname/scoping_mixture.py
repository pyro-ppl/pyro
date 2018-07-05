from __future__ import absolute_import, division, print_function

import argparse
import torch
from torch.distributions import constraints

import pyro
import pyro.optim
import pyro.distributions as dist

from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO

from pyro.contrib.autoname import scope


def model(K, data):
    # Global parameters.
    weights = pyro.param('weights', torch.ones(K) / K, constraint=constraints.simplex)
    locs = pyro.param('locs', 10 * torch.randn(K))
    scale = pyro.param('scale', torch.tensor(0.5), constraint=constraints.positive)

    with pyro.iarange('data'):
        return local_model(weights, locs, scale, data)


@scope(prefix="local")
def local_model(weights, locs, scale, data):
    assignment = pyro.sample('assignment',
                             dist.Categorical(weights).expand_by([len(data)]))
    return pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)


def guide(K, data):
    assignment_probs = pyro.param('assignment_probs', torch.ones(len(data), K) / K,
                                  constraint=constraints.unit_interval)
    with pyro.iarange('data'):
        return local_guide(assignment_probs)


@scope(prefix="local")
def local_guide(probs):
    return pyro.sample('assignment', dist.Categorical(probs))


def main(args):
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    K = 2

    data = torch.tensor([0.0, 1.0, 2.0, 20.0, 30.0, 40.0])
    optim = pyro.optim.Adam({'lr': 0.1})
    inference = SVI(model, config_enumerate(guide, 'parallel'), optim,
                    loss=TraceEnum_ELBO(max_iarange_nesting=1))

    print('Step\tLoss')
    loss = 0.0
    for step in range(args.num_epochs):
        if step and step % 10 == 0:
            print('{}\t{:0.5g}'.format(step, loss))
            loss = 0.0
        loss += inference.step(K, data)

    print('Parameters:')
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        print('{} = {}'.format(name, pyro.param(name).detach().cpu().numpy()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=200, type=int)
    args = parser.parse_args()
    main(args)
