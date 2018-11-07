# a state space model with multiple importance sampling proposal
#
# to get this to work we need to make guide-side enumeration work
# inside of the log

from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam

from pyro.infer.collapse import collapse

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


def model(sequences, args, batch_size=None):
    zs = []
    with pyro.plate("sequences", args.num_sequences, batch_size) as batch:
        y = 0.0
        for t in range(args.length):
            y = pyro.sample("y_{}".format(t), dist.Normal(y, 1.0))
            obs_t = None if sequences is None else sequences[batch, t]
            z = pyro.sample("z_{}".format(t), dist.Normal(y.abs(), args.sigma_obs),
                            obs=obs_t)
            zs.append(z)

    return torch.stack(zs, dim=-1)


def guide(sequences, args, batch_size=None):
    px_shape = torch.Size((args.length, args.num_sequences, args.hidden_dim,))
    probs_x = pyro.param("probs_x", torch.ones(px_shape), constraint=constraints.simplex)

    y_shape = torch.Size((args.length, args.hidden_dim, args.num_sequences))
    loc_y = pyro.param("loc", 0.03 * torch.randn(y_shape))
    scale_y = pyro.param("scale", (0.03 * torch.randn(y_shape)).exp(), constraint=constraints.positive)
    with pyro.plate("sequences", args.num_sequences, batch_size) as batch:
        for t in range(args.length):
            x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[t][batch]),
                    infer={"enumerate": "parallel", "collapse": True})
            # print('probx', dist.Categorical(probs_x[t][batch]).shape())
            # print('x', x.shape)
            # print('locy', (loc_y[t][x, batch].shape))
            y = pyro.sample("y_{}".format(t),
                dist.Normal(loc_y[t][x, batch], scale_y[t][x, batch]))
                # infer={"num_samples": args.num_samples})
            # print('y', y.shape)


def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('-' * 40)

    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    sequences = model(None, args)

    elbo = TraceEnum_ELBO(max_plate_nesting=1, strict_enumeration_warning=False)
    optim = Adam({'lr': args.learning_rate})

    collapsed_guide = collapse(guide, 1)
    svi = SVI(model, collapsed_guide, optim, elbo)

    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(sequences, args, batch_size=args.batch_size)
        logging.info('{: >5d}\t{}'.format(step, loss / sequences.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MIS SSM")
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-b", "--batch-size", default=10, type=int)
    parser.add_argument("-d", "--hidden-dim", default=2, type=int)
    parser.add_argument("-sig", "--sigma-obs", default=0.2, type=float)
    parser.add_argument("-nseq", "--num-sequences", default=100, type=int)
    parser.add_argument("-nsamp", "--num-samples", default=3, type=int)
    parser.add_argument("-l", "--length", default=20, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    main(args)
