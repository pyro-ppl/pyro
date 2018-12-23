from __future__ import absolute_import, division, print_function

import uuid
import argparse
import time
import os

import math
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


def main(**args):
    N_train = {'jsb': 229, 'piano': 87, 'nottingham': 694, 'muse': 524}
    if args['dataset'] == 'jsb':
        args['batch_size'] = 20
        args['num_steps'] = 300
        NN = args['num_steps'] * N_train[args['dataset']] / args['batch_size']
        args['learning_rate_decay'] = math.exp(math.log(10.0e-3) / NN)
    elif args['dataset'] == 'piano':
        args['batch_size'] = 15
        args['num_steps'] = 300
        NN = args['num_steps'] * N_train[args['dataset']] / args['batch_size']
        args['learning_rate_decay'] = math.exp(math.log(10.0e-3) / NN)
    elif args['dataset'] in ['muse', 'nottingham']:
        args['batch_size'] = 30
        args['num_steps'] = 100
        NN = args['num_steps'] * N_train[args['dataset']] / args['batch_size']
        args['learning_rate_decay'] = math.exp(math.log(10.0e-3) / NN)

    log_tag = 'hmm.{}.model{}.num_steps_{}.bs_{}.hd_{}.seed_{}.b1_{:.3f}'
    log_tag += '.lrd_{:.5f}.lr_{:.3f}.cn_{:.1f}'
    log_tag = log_tag.format(args['dataset'], args['model'], args['num_steps'], args['batch_size'],
                             args['hidden_dim'], args['seed'], args['beta1'], args['learning_rate_decay'],
                             args['learning_rate'], args['clip_norm'])

    uid = str(uuid.uuid4())[0:4]
    log = get_logger(args['log_dir'], log_tag + '.' + uid + '.log')
    log(args)

    data_dim = 88

    if args['cuda']:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    log('Loading data')
    data = {"jsb": poly.JSB_CHORALES,
            "piano": poly.PIANO_MIDI,
            "muse": poly.MUSE_DATA,
            "nottingham": poly.NOTTINGHAM}
    data = poly.load_data(data[args['dataset']])

    log('-' * 40)
    model = models[args['model']](args, data_dim)

    log('Training model{} on {} sequences ({} test sequences)'.format(
                 args['model'], len(data['train']['sequences']), len(data['test']['sequences'])))
    train_sequences = data['train']['sequences'].float()
    train_lengths = data['train']['sequence_lengths'].long()
    test_sequences = data['test']['sequences'].float()
    test_lengths = data['test']['sequence_lengths'].long()
    log("Average sequence lengths: %d (train), %d (test)" % (train_lengths.float().mean(), test_lengths.float().mean()))

    N_train_obs = float(train_lengths.sum())
    N_test_obs = float(test_lengths.sum())

    pyro.set_rng_seed(args['seed'])
    pyro.clear_param_store()

    #guide = AutoDelta(poutine.block(model.model,
    #                  expose_fn=lambda msg: msg["name"].startswith("probs_")))
    def guide(*args, **kwargs):
        pass

    Elbo = JitTraceEnum_ELBO if args['jit'] else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=2)
    optim = ClippedAdam({'lr': args['learning_rate'], 'betas': (args['beta1'], 0.999),
                         'lrd': args['learning_rate_decay'], 'clip_norm': args['clip_norm']})
    svi = SVI(model.model, guide, optim, elbo)

    ts = [time.time()]
    log('Step\tLoss\tEpoch Time')

    def evaluate(sequences, lengths, evaluate_batch_size=50):
        mb_indices = get_mb_indices(sequences.size(0), evaluate_batch_size)
        loss = 0.0

        for mb in mb_indices:
            loss += svi.evaluate_loss(sequences, lengths, args, mb=mb)

        return loss

    best_train_loss = 9.9e9
    best_test_loss = 9.9e9

    for epoch in range(args['num_steps']):
        epoch_loss = 0.0
        mb_indices = get_mb_indices(train_sequences.size(0), args['batch_size'])
        for mb in mb_indices:
            epoch_loss += svi.step(train_sequences, train_lengths, args, mb=mb)
            store = pyro.get_param_store()
            store["probs_x"] = store["probs_x"].clamp(args['clamp_prob'])
            if 'probs_y' in store:
                store["probs_y"] = store["probs_y"].clamp(args['clamp_prob'])
            if 'probs_w' in store:
                store["probs_w"] = store["probs_w"].clamp(args['clamp_prob'])

        ts.append(time.time())
        log('{: >5d}\t{:.4f}\t{:.2f}'.format(epoch, epoch_loss / N_train_obs,
                                                      (ts[-1] - ts[0])/(epoch+1)))

        if epoch > 0 and (epoch % args['eval_frequency'] == 0 or epoch == args['num_steps'] - 1):
            train_loss = evaluate(train_sequences, train_lengths) / N_train_obs
            test_loss = evaluate(test_sequences, test_lengths) / N_test_obs
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_test_loss = test_loss
            log('{: >5d}\ttraining loss = {:.5f}'.format(epoch, train_loss))
            log('{: >5d}\ttest loss = {:.5f}'.format(epoch, test_loss))

    log('-' * 40)

    capacity = sum(len(pyro.param(name).reshape(-1))
                   for name in pyro.get_param_store().get_all_param_names())
    log('model{} capacity = {} parameters'.format(args['model'], capacity))
    print("best_train_loss: {:.4f}  best_test_loss: {:.4f}".format(best_train_loss, best_test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMM variants")
    parser.add_argument("-ld", "--log-dir", default="hmm_logs/", type=str)
    parser.add_argument("-d", "--dataset", default="jsb", type=str)
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=300, type=int)
    parser.add_argument("-b", "--batch-size", default=20, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=8, type=int)
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("-b1", "--beta1", default=0.9, type=float)
    parser.add_argument("-cp", "--clamp-prob", default=1.0e-12, type=float)
    parser.add_argument("-cn", "--clip-norm", default=20.0, type=float)
    parser.add_argument("-lrd", "--learning_rate_decay", default=0.999, type=float)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-ef", "--eval-frequency", default=1, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')

    args = parser.parse_args()
    main(**vars(args))
