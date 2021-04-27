import argparse
import functools
import logging
import sys

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.contrib.examples import polyphonic_data_loader as poly
from pyro.infer.autoguide import AutoDelta
from pyro.ops.indexing import Vindex
from pyro.util import ignore_jit_warnings

import pyro.contrib.funsor
from pyroapi import distributions as dist
from pyroapi import handlers, infer, optim, pyro, pyro_backend

import funsor
from utils import get_mb_indices


logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


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
        with pyro.plate("sequences", mb.size(0), dim=-3), handlers.scale(scale=args.scale), \
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
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')
    data = poly.load_data(poly.JSB_CHORALES)
    train_sequences = data['train']['sequences'].float()
    train_lengths = data['train']['sequence_lengths'].long()
    test_sequences = data['test']['sequences'].float()
    test_lengths = data['test']['sequence_lengths'].long()
    data_dim = 88

    logging.info('-' * 40)
    model = models[args.model](args, data_dim)

    logging.info('Training {} on {} sequences'.format("model{}".format(args.model), len(train_sequences)))

    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    def guide(*args, **kwargs):
        pass

    N_train_obs = float(train_lengths.sum())
    N_obs_mb = N_train_obs * args.batch_size / train_sequences.size(0)
    args.scale = 1.0 / N_obs_mb if args.scale_loss else 1.0

    model = functools.partial(model.model, args)
    guide = functools.partial(guide, args=args)

    optimizer = optim.Adam({'lr': args.learning_rate})

    Elbo = infer.JitTraceMarkovEnum_ELBO
    max_plate_nesting = 3
    elbo = Elbo(max_plate_nesting=max_plate_nesting,
                strict_enumeration_warning=True,
                jit_options={"time_compilation": args.time_compilation})
    svi = infer.SVI(model, guide, optimizer, elbo)

    logging.info('Step\tLoss')
    for epoch in range(args.num_steps):
        epoch_loss = 0.0
        mb_indices, masks = get_mb_indices(train_sequences.size(0), args.batch_size)

        for mb, mask in zip(mb_indices, masks):
            epoch_loss += svi.step(train_sequences, train_lengths, mb, mask)

        logging.info('{: >5d}\t{}'.format(epoch, epoch_loss))

    if args.time_compilation:
        logging.debug('time to compile: {} s.'.format(elbo._differentiable_loss.compile_time))

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
    assert pyro.__version__.startswith('1.6.0')
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=20, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--scale-loss', action='store_true')
    parser.add_argument('--time-compilation', action='store_true')
    args = parser.parse_args()

    funsor.set_backend("torch")
    PYRO_BACKEND = "contrib.funsor"

    with pyro_backend(PYRO_BACKEND):
        main(args)
