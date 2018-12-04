from __future__ import absolute_import, division, print_function

import argparse
import time
import os

import torch
import torch.nn as nn

import dmm.polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam, ClippedAdam
from models_hmm import model1, model2, model3, model4, model5
from utils_hmm import get_mb_indices, get_logger

pyro.distributions.enable_validation(False)

models = {name[len('model'):]: model
              for name, model in globals().items()
              if name.startswith('model')}


def main(args):
    log_tag = 'hmm.model{}.num_steps_{}.bs_{}.hd_{}.seed_{}.b1_{:.3f}'
    log_tag += '.lrd_{:.5f}.lr_{:.3f}.cn_{:.1f}'
    log_tag = log_tag.format(args.model, args.num_steps, args.batch_size,
                             args.hidden_dim, args.seed, args.beta1, args.learning_rate_decay,
                             args.learning_rate, args.clip_norm)
    if not os.path.exists('hmm_logs'):
        os.makedirs('hmm_logs')
    log = get_logger('hmm_logs/', log_tag + '.log')
    log(args)

    data_dim = 88

    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    log('Loading data')
    data = poly.load_data()

    log('-' * 40)
    model = models[args.model](args, data_dim)

    log('Training model{} on {} sequences'.format(
                 args.model, len(data['train']['sequences'])))
    sequences = torch.tensor(data['train']['sequences'], dtype=torch.float32)
    lengths = torch.tensor(data['train']['sequence_lengths'], dtype=torch.long)
    if args.truncate:
        lengths.clamp_(max=args.truncate)
    num_observations = float(lengths.sum())

    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    guide = AutoDelta(poutine.block(model.model,
                      expose_fn=lambda msg: msg["name"].startswith("probs_")))

    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=2)
    optim = ClippedAdam({'lr': args.learning_rate, 'betas': (args.beta1, 0.999),
                         'lrd': args.learning_rate_decay, 'clip_norm': args.clip_norm})
    svi = SVI(model.model, guide, optim, elbo)

    ts = [time.time()]
    log('Step\tLoss\tEpoch Time')

    for epoch in range(args.num_steps):
        epoch_loss = 0.0
        mb_indices = get_mb_indices(sequences.size(0), args.batch_size)

        for mb in mb_indices:
            epoch_loss += svi.step(sequences, lengths, args, mb=mb)

        ts.append(time.time())
        log('{: >5d}\t{:.4f}\t{:.2f}'.format(epoch, epoch_loss / num_observations,
                                                      (ts[-1] - ts[0])/(epoch+1)))

    train_loss = elbo.loss(model.model, guide, sequences, lengths, args, include_prior=False)
    log('training loss = {}'.format(train_loss / num_observations))

    log('-' * 40)
    log('Evaluating on {} test sequences'.format(len(data['test']['sequences'])))
    sequences = torch.tensor(data['test']['sequences'], dtype=torch.float32)
    lengths = torch.tensor(data['test']['sequence_lengths'], dtype=torch.long)
    if args.truncate:
        lengths.clamp_(max=args.truncate)
    num_observations = float(lengths.sum())
    test_loss = elbo.loss(model.model, guide, sequences, lengths, args, include_prior=False)
    log('test loss = {}'.format(test_loss / num_observations))

    capacity = sum(len(pyro.param(name).reshape(-1))
                   for name in pyro.get_param_store().get_all_param_names())
    log('model{} capacity = {} parameters'.format(args.model, capacity))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMM variants")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=300, type=int)
    parser.add_argument("-b", "--batch-size", default=20, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("-b1", "--beta1", default=0.8, type=float)
    parser.add_argument("-cn", "--clip-norm", default=20.0, type=float)
    parser.add_argument("-lrd", "--learning_rate_decay", default=0.999, type=float)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
