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


logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)



def model_7(sequences, lengths, args, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    with handlers.mask(mask=include_prior):
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, data_dim])
                                  .to_event(2))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", num_sequences, batch_size, dim=-3) as batch:
        lengths = lengths[batch]
        batch = batch[:, None]
        x_prev = 0
        for t in pyro.vectorized_markov(name="time", size=int(max_length if args.jit else lengths.max()), dim=-2):
            with handlers.mask(mask=(t < lengths.unsqueeze(-1)).unsqueeze(-1)):
                x_curr = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x_prev]),
                                     infer={"enumerate": "parallel"})
                with tones_plate:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[x_curr.squeeze(-1)]),
                                obs=Vindex(sequences)[batch, t])


models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}


def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')
    data = poly.load_data(poly.JSB_CHORALES)

    logging.info('-' * 40)
    model = models[args.model]
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(data['train']['sequences'])))
    sequences = data['train']['sequences']
    lengths = data['train']['sequence_lengths']

    # find all the notes that are present at least once in the training set
    present_notes = ((sequences == 1).sum(0).sum(0) > 0)
    # remove notes that are never played (we remove 37/88 notes)
    sequences = sequences[..., present_notes]

    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
        sequences = sequences[:, :args.truncate]
    num_observations = float(lengths.sum())
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    guide = AutoDelta(handlers.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    if args.print_shapes:
        if args.model == "0":
            first_available_dim = -2
        elif args.model == "7":
            first_available_dim = -4
        else:
            first_available_dim = -3
        guide_trace = handlers.trace(guide).get_trace(
            sequences, lengths, args=args, batch_size=args.batch_size)
        model_trace = handlers.trace(
            handlers.replay(handlers.enum(model, first_available_dim), guide_trace)).get_trace(
            sequences, lengths, args=args, batch_size=args.batch_size)
        logging.info(model_trace.format_shapes())

    # Bind non-PyTorch parameters to make these functions jittable.
    model = functools.partial(model, args=args)
    guide = functools.partial(guide, args=args)

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    optimizer = optim.Adam({'lr': args.learning_rate})

    Elbo = infer.JitTraceMarkovEnum_ELBO if args.jit else infer.TraceMarkovEnum_ELBO
    max_plate_nesting = 3
    elbo = Elbo(max_plate_nesting=max_plate_nesting,
                strict_enumeration_warning=True,
                jit_options={"time_compilation": args.time_compilation})
    svi = infer.SVI(model, guide, optimizer, elbo)

    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(sequences, lengths, batch_size=args.batch_size)
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))

    if args.jit and args.time_compilation:
        logging.debug('time to compile: {} s.'.format(elbo._differentiable_loss.compile_time))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, sequences, lengths, batch_size=sequences.shape[0], include_prior=False)
    logging.info('training loss = {}'.format(train_loss / num_observations))

    # Finally we evaluate on the test dataset.
    logging.info('-' * 40)
    logging.info('Evaluating on {} test sequences'.format(len(data['test']['sequences'])))
    sequences = data['test']['sequences'][..., present_notes]
    lengths = data['test']['sequence_lengths']
    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
    num_observations = float(lengths.sum())

    test_loss = elbo.loss(model, guide, sequences, lengths, batch_size=sequences.shape[0], include_prior=False)
    logging.info('test loss = {}'.format(test_loss / num_observations))

    capacity = sum(value.reshape(-1).size(0)
                   for value in pyro.get_param_store().values())
    logging.info('model_{} capacity = {} parameters'.format(args.model, capacity))


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.6.0')
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--time-compilation', action='store_true')
    parser.add_argument('-rp', '--raftery-parameterization', action='store_true')
    parser.add_argument('--tmc', action='store_true',
                        help="Use Tensor Monte Carlo instead of exact enumeration "
                             "to estimate the marginal likelihood. You probably don't want to do this, "
                             "except to see that TMC makes Monte Carlo gradient estimation feasible "
                             "even with very large numbers of non-reparametrized variables.")
    parser.add_argument('--tmc-num-samples', default=10, type=int)
    parser.add_argument('--funsor', action='store_true')
    args = parser.parse_args()

    if args.funsor:
        import funsor
        funsor.set_backend("torch")
        PYRO_BACKEND = "contrib.funsor"
    else:
        PYRO_BACKEND = "pyro"

    with pyro_backend(PYRO_BACKEND):
        main(args)
