from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch

import dmm.polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.ops.einsum import cached_paths
from pyro.optim import Adam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


def model(sequences, sequence_lengths, trans_prior, emit_prior, args):
    assert len(sequences) == len(sequence_lengths)
    trans = pyro.sample("trans", trans_prior)
    emit = pyro.sample("emit", emit_prior)
    tones_iarange = pyro.iarange("tones", sequences.shape[-1], dim=-1)
    with pyro.iarange("sequences", len(sequences), args.batch_size, dim=-2) as batch:
        lengths = sequence_lengths[batch]
        x = 0
        for t in range(lengths.max()):
            with poutine.mask(mask=(lengths > t).unsqueeze(-1)):
                x = pyro.sample("x_{}".format(t), dist.Categorical(trans[x]),
                                infer={"enumerate": "parallel", "expand": False})
                with tones_iarange:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(emit[x]),
                                obs=sequences[batch, t])


def main(args):
    logging.info('Loading data')
    data = poly.load_data()
    sequences = torch.tensor(data['train']['sequences'], dtype=torch.float32)
    sequence_lengths = torch.tensor(data['train']['sequence_lengths'], dtype=torch.long)
    data_dim = sequences.shape[-1]

    logging.info('Initialize')
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)
    trans_prior = dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1).independent(1)
    emit_prior = dist.Beta(0.5 * torch.ones(args.hidden_dim, data_dim),
                           0.5 * torch.ones(args.hidden_dim, data_dim)).independent(2)
    pyro.param('auto_trans',
               0.8 * torch.eye(args.hidden_dim) + 0.1 + 0.1 * trans_prior.sample(),
               constraint=trans_prior.support)
    pyro.param('auto_emit',
               0.9 * 0.5 * torch.ones(args.hidden_dim, data_dim) + 0.1 * emit_prior.sample(),
               constraint=emit_prior.support)
    assert pyro.param('auto_trans').shape == (args.hidden_dim, args.hidden_dim)
    assert pyro.param('auto_emit').shape == (args.hidden_dim, data_dim)

    logging.info('Training on {} sequences'.format(len(sequences)))
    guide = AutoDelta(poutine.block(model, expose=["trans", "emit"]))
    optim = Adam({'lr': 0.1})
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_iarange_nesting=2)
    svi = SVI(model, guide, optim, elbo)

    logging.info('Epoch\tLoss')
    num_observations = float(sequence_lengths.sum())
    with cached_paths('data/opt_einsum_path_cache.pkl'):
        for epoch in range(args.num_epochs):
            loss = svi.step(sequences, sequence_lengths, trans_prior, emit_prior, args)
            logging.info('{: >5d}\t{}'.format(epoch, loss / num_observations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayesian Baum-Welch learning Bach Chorales")
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
