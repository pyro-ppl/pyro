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


def model(sequences, lengths, args, batch_size=None):
    assert sequences.shape[0] == lengths.shape[0]
    assert sequences.shape[1] >= lengths.max()
    data_dim = sequences.shape[-1]
    trans = pyro.sample("p_trans",
                        dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                            .independent(1))
    emit = pyro.sample("p_emit",
                       dist.Beta(0.1, 0.9)
                           .expand([args.hidden_dim, data_dim]).independent(2))
    tones_iarange = pyro.iarange("tones", sequences.shape[-1], dim=-1)
    with pyro.iarange("sequences", len(sequences), batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        for t in range(lengths.max()):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                x = pyro.sample("x_{}".format(t), dist.Categorical(trans[x]),
                                infer={"enumerate": "parallel", "expand": False})
                with tones_iarange:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(emit[x]),
                                obs=sequences[batch, t])


def main(args):
    logging.info('Loading data')
    data = poly.load_data()
    with cached_paths('data/opt_einsum_path_cache.pkl'):

        logging.info('-' * 40)
        logging.info('Training on {} sequences'.format(len(data['train']['sequences'])))
        sequences = torch.tensor(data['train']['sequences'], dtype=torch.float32)
        lengths = torch.tensor(data['train']['sequence_lengths'], dtype=torch.long)
        num_observations = float(lengths.sum())
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        pyro.enable_validation(True)
        guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("p_")))
        optim = Adam({'lr': 0.1})
        Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
        elbo = Elbo(max_iarange_nesting=2)
        svi = SVI(model, guide, optim, elbo)
        logging.info('Epoch\tLoss')
        try:
            for epoch in range(args.num_epochs):
                loss = svi.step(sequences, lengths, args, batch_size=args.batch_size)
                logging.info('{: >5d}\t{}'.format(epoch, loss / num_observations))
        except KeyboardInterrupt:
            pass
        train_loss = elbo.loss(model, guide, sequences, lengths, args) / num_observations
        logging.info('training loss = {}'.format(train_loss))

        logging.info('-' * 40)
        logging.info('Evaluating on {} test sequences'.format(len(data['test']['sequences'])))
        sequences = torch.tensor(data['test']['sequences'], dtype=torch.float32)
        lengths = torch.tensor(data['test']['sequence_lengths'], dtype=torch.long)
        num_observations = float(lengths.sum())
        test_loss = elbo.loss(model, guide, sequences, lengths, args) / num_observations
        logging.info('test loss = {}'.format(test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayesian Baum-Welch learning Bach Chorales")
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
