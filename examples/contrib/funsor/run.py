import argparse
import time
import math
import functools
import logging
import sys
from copy import copy

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.contrib.examples import polyphonic_data_loader as poly
from pyro.infer.autoguide import AutoDelta
from pyro.ops.indexing import Vindex
from pyro.util import ignore_jit_warnings
from pyro.optim import ClippedAdam

import pyro.contrib.funsor
from pyroapi import distributions as dist
from pyroapi import handlers, infer, optim, pyro, pyro_backend

import funsor
from utils import get_mb_indices, get_logger
import uuid


# HMM
class model1(nn.Module):
    def __init__(self, args, data_dim):
        super(model1, self).__init__()

    @ignore_jit_warnings()
    def model(self, args, sequences, lengths, mb, mask):
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args.hidden_dim
        probs_x = pyro.param("probs_x", lambda: torch.rand(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_y = pyro.param("probs_y", lambda: torch.rand(hidden_dim, data_dim),
                             constraint=constraints.unit_interval)

        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", mb.size(0), dim=-3), handlers.scale(scale=torch.tensor(args.scale)), \
            handlers.mask(mask=mask.unsqueeze(-1).unsqueeze(-1)):
            lengths = lengths[mb]
            x_prev = 0
            for t in pyro.vectorized_markov(name="time", size=max_length, dim=-2):
                with handlers.mask(mask=(t < lengths.unsqueeze(-1)).unsqueeze(-1)):
                    x_curr = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x_prev]),
                                         infer={"enumerate": "parallel"})
                    with tones_plate:
                        pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[x_curr.squeeze(-1)]),
                                    obs=Vindex(sequences)[mb.unsqueeze(-1), t])


models = {name[len('model'):]: model
          for name, model in globals().items()
          if name.startswith('model')}


def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    N_train = {'jsb': 229, 'piano': 87, 'nottingham': 694, 'muse': 524}[args.dataset]
    if args.dataset == 'jsb':
        args.batch_size = 20
        args.num_steps = 300 # 400
        NN = args.num_steps * N_train / args.batch_size
        args.learning_rate_decay = math.exp(math.log(args.learning_rate_decay) / NN)
    elif args.dataset == 'piano':
        args.batch_size = 15
        args.num_steps = 300 # 400
        NN = args.num_steps * N_train / args.batch_size
        args.learning_rate_decay = math.exp(math.log(args.learning_rate_decay) / NN)
    elif args.dataset == 'nottingham':
        args.batch_size = 30
        args.num_steps = 200
        NN = args.num_steps * N_train / args.batch_size
        args.learning_rate_decay = math.exp(math.log(args.learning_rate_decay) / NN)
    elif args.dataset == 'muse':
        args.batch_size = 20
        args.num_steps = 150
        NN = args.num_steps * N_train / args.batch_size
        args.learning_rate_decay = math.exp(math.log(args.learning_rate_decay) / NN)

    log_tag = 'hmm.{}.model{}.num_steps_{}.bs_{}.hd_{}.seed_{}'
    log_tag += '.lrd_{:.5f}.lr_{:.3f}.nn_{}_{}.scale_{}'
    log_tag = log_tag.format(args.dataset, args.model, args.num_steps, args.batch_size,
                             args.hidden_dim, args.seed, args.learning_rate_decay,
                             args.learning_rate, args.nn_dim, args.nn_channels,
                             args.scale_loss)

    uid = str(uuid.uuid4())[0:4]
    log = get_logger('./logs/', log_tag + '.' + uid + '.log')
    log(args)

    data = poly.load_data(poly.JSB_CHORALES)
    train_sequences = data['train']['sequences'].float()
    train_lengths = data['train']['sequence_lengths'].long()
    test_sequences = data['test']['sequences'].float()
    test_lengths = data['test']['sequence_lengths'].long()
    data_dim = 88

    model = models[args.model](args, data_dim)

    log('Training {} on {} sequences'.format("model{}".format(args.model), len(train_sequences)))

    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    def guide(*args, **kwargs):
        pass

    N_train_obs = float(train_lengths.sum())
    N_obs_mb = N_train_obs * args.batch_size / train_sequences.size(0)
    N_test_obs = float(test_lengths.sum())

    model_eval = functools.partial(model.model, copy(args))
    args.scale = 1.0 / N_obs_mb if args.scale_loss else 1.0
    model_train = functools.partial(model.model, copy(args))
    guide = functools.partial(guide, args=args)

    optim = ClippedAdam({'lr': args.learning_rate, 'betas': (0.90, 0.999),
                         'lrd': args.learning_rate_decay, 'clip_norm': 20.0})

    Elbo = infer.JitTraceMarkovEnum_ELBO
    max_plate_nesting = 3
    elbo = Elbo(max_plate_nesting=max_plate_nesting, strict_enumeration_warning=True)
    elbo_eval = Elbo(max_plate_nesting=max_plate_nesting, strict_enumeration_warning=True)
    svi = infer.SVI(model_train, guide, optim, elbo)

    @torch.no_grad()
    def evaluate(sequences, lengths, evaluate_batch_size=8):
        mb_indices, masks = get_mb_indices(sequences.size(0), evaluate_batch_size)
        return sum([elbo_eval.differentiable_loss(model_eval, guide, sequences, lengths, mb, mask)
                    for mb, mask in zip(mb_indices, masks)])

    ts = [time.time()]
    eval_frequency = 5

    for epoch in range(args.num_steps):
        epoch_loss = 0.0
        mb_indices, masks = get_mb_indices(train_sequences.size(0), args.batch_size)

        for mb, mask in zip(mb_indices, masks):
            epoch_loss += svi.step(train_sequences, train_lengths, mb, mask)

        ts.append(time.time())
        log('[epoch %03d]  running train loss: %.4f\t\t (epoch dt: %.2f)' % (epoch, epoch_loss,
                                                      (ts[-1] - ts[0])/(epoch+1)))
        if epoch > 3 and epoch % eval_frequency == 0:
            train_loss = evaluate(train_sequences, train_lengths) / N_train_obs
            test_loss = evaluate(test_sequences, test_lengths) / N_test_obs
            log('[epoch %03d]  train loss: %.4f' % (epoch, train_loss))
            log('[epoch %03d]   test loss: %.4f' % (epoch, test_loss))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    #train_loss = elbo.loss(model, guide, sequences, lengths, batch_size=sequences.shape[0], include_prior=False)
    #logging.info('training loss = {}'.format(train_loss / num_observations))

    # Finally we evaluate on the test dataset.
    #logging.info('-' * 40)
    #logging.info('Evaluating on {} test sequences'.format(len(data['test']['sequences'])))

    #test_loss = elbo.loss(model, guide, sequences, lengths, batch_size=sequences.shape[0], include_prior=False)
    #logging.info('test loss = {}'.format(test_loss / num_observations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    parser.add_argument("--dataset", default="jsb", type=str)
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=20, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-d", "--hidden-dim", default=36, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.03, type=float)
    parser.add_argument("--scale", default=1.0, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=3.0e-5, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--scale-loss', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    funsor.set_backend("torch")
    PYRO_BACKEND = "contrib.funsor"

    with pyro_backend(PYRO_BACKEND):
        main(args)
