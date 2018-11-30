"""
This example shows how to marginalize out discrete model variables in Pyro.

This combines Stochastic Variational Inference (SVI) with an
Expectation-Maximization (EM) algorithm, where we use enumeration to exactly
marginalize out some variables from the ELBO computation.

To marginalize out discrete variables ``x`` in Pyro's SVI:
1. Verify that the variable dependency structure in your model
    admits tractable inference, i.e. the dependency graph among
    enumerated variables should have narrow treewidth.
2. Annotate each target each such sample site in the model
    with ``infer={"enumerate": "parallel"}``
3. Ensure your model can handle broadcasting of the sample values
    of those variables
4. Use the ``TraceEnum_ELBO`` loss inside Pyro's ``SVI``.
"""
from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
import torch.nn as nn

import dmm.polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


# Let's start with a simple Hidden Markov Model.
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
#
# This model includes two plates: one for minibatches of data, and one
# for the data_dim = 88 keys on the piano. This model has two "style" parameters
# probs_x and probs_y that we'll draw from a prior. The latent state is x,
# and the observed state is y. We'll drive probs_* with the guide, enumerate
# over x, and condition on y.
#
# Importantly, the dependency structure of the enumerated variables has
# narrow treewidth, therefore admitting efficient inference by message passing.
# Pyro's TraceEnum_ELBO will find an efficient message passing scheme if one
# exists.
def model_1(sequences, lengths, args, batch_size=None, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    assert lengths.shape == (num_sequences,)
    assert lengths.max() <= max_length
    with poutine.mask(mask=include_prior):
        # Our prior on transition probabilities will be:
        # stay in the same state with 90% probability; uniformly jump to another
        # state with 10% probability.
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                  .independent(1))
        # We put a weak prior on the conditional probability of a tone sounding.
        # We know that on average about 4 of 88 tones are active, so we'll set a
        # rough weak prior of 10% of the notes being active at any one time.
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, data_dim])
                                  .independent(2))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", len(sequences), batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        for t in pyro.markov(range(lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                # On the next line, we'll overwrite the value of x with an updated
                # value. If we wanted to record all x values, we could instead
                # write x[t] = pyro.sample(...x[t-1]...).
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                with tones_plate:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[x.squeeze(-1)]),
                                obs=sequences[batch, t])


# Next let's add a dependency of y[t] on y[t-1].
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1] --> y[t] --> y[t+1]
#
def model_2(sequences, lengths, args, batch_size=None, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    assert lengths.shape == (num_sequences,)
    assert lengths.max() <= max_length
    with poutine.mask(mask=include_prior):
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                  .independent(1))
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, 2, data_dim])
                                  .independent(3))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", len(sequences), batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x, y = 0, 0
        for t in pyro.markov(range(lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                # Note the broadcasting tricks here: to index probs_y on tensors x and y,
                # we also need a final tensor for the tones dimension. This is conveniently
                # provided by the plate associated with that dimension.
                with tones_plate as tones:
                    y = pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[x, y, tones]),
                                    obs=sequences[batch, t]).long()


# Next consider a Factorial HMM with two hidden states.
#
#    w[t-1] ----> w[t] ---> w[t+1]
#        \ x[t-1] --\-> x[t] --\-> x[t+1]
#         \  /       \  /       \  /
#          \/         \/         \/
#        y[t-1]      y[t]      y[t+1]
#
# Note that since the joint distribution of each y[t] depends on two variables,
# those two variables become dependent. Therefore during enumeration, the
# entire joint space of these variables w[t],x[t] needs to be enumerated.
# For that reason, we set the dimension of each to the square root of the
# target hidden dimension.
def model_3(sequences, lengths, args, batch_size=None, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    assert lengths.shape == (num_sequences,)
    assert lengths.max() <= max_length
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    with poutine.mask(mask=include_prior):
        probs_w = pyro.sample("probs_w",
                              dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1)
                                  .independent(1))
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1)
                                  .independent(1))
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([hidden_dim, hidden_dim, data_dim])
                                  .independent(3))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", len(sequences), batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        w, x = 0, 0
        for t in pyro.markov(range(lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                w = pyro.sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                infer={"enumerate": "parallel"})
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                with tones_plate as tones:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[w, x, tones]),
                                obs=sequences[batch, t])


# By adding a dependency of x on w, we generalize to a
# Dynamic Bayesian Network.
#
#     w[t-1] ----> w[t] ---> w[t+1]
#        |  \       |  \       |   \
#        | x[t-1] ----> x[t] ----> x[t+1]
#        |   /      |   /      |   /
#        V  /       V  /       V  /
#     y[t-1]       y[t]      y[t+1]
#
# Note that message passing here has roughly the same cost as with the
# Factorial HMM, but this model has more parameters.
def model_4(sequences, lengths, args, batch_size=None, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    assert lengths.shape == (num_sequences,)
    assert lengths.max() <= max_length
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    hidden = torch.arange(hidden_dim, dtype=torch.long)
    with poutine.mask(mask=include_prior):
        probs_w = pyro.sample("probs_w",
                              dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1)
                                  .independent(1))
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1)
                                  .expand_by([hidden_dim])
                                  .independent(2))
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([hidden_dim, hidden_dim, data_dim])
                                  .independent(3))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", len(sequences), batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        # Note the broadcasting tricks here: we declare a hidden torch.arange and
        # ensure that w and x are always tensors so we can unsqueeze them below,
        # thus ensuring that the x sample sites have correct distribution shape.
        w = x = torch.tensor(0, dtype=torch.long)
        for t in pyro.markov(range(lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                w = pyro.sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                infer={"enumerate": "parallel"})
                x = pyro.sample("x_{}".format(t),
                                dist.Categorical(probs_x[w.unsqueeze(-1), x.unsqueeze(-1), hidden]),
                                infer={"enumerate": "parallel"})
                with tones_plate as tones:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[w, x, tones]),
                                obs=sequences[batch, t])


# Next let's consider a neural HMM model.
#
#     x[t-1] --> x[t] --> x[t+1]   } standard HMM +
#        |        |         |
#        V        V         V
#     y[t-1] --> y[t] --> y[t+1]   } neural likelihood
#
# First let's define a neural net to generate y logits.
class TonesGenerator(nn.Module):
    def __init__(self, args, data_dim):
        self.args = args
        self.data_dim = data_dim
        super(TonesGenerator, self).__init__()
        self.x_to_hidden = nn.Linear(args.hidden_dim, args.nn_dim)
        self.y_to_hidden = nn.Linear(args.nn_channels * data_dim, args.nn_dim)
        self.conv = nn.Conv1d(1, args.nn_channels, 3, padding=1)
        self.hidden_to_logits = nn.Linear(args.nn_dim, data_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # Check dimension of y so this can be used with and without enumeration.
        if y.dim() < 2:
            y = y.unsqueeze(0)

        # Hidden units depend on two inputs: a one-hot encoded categorical variable x, and
        # a bernoulli variable y. Whereas x will typically be enumerated, y will be observed.
        # We apply x_to_hidden independently from y_to_hidden, then broadcast the non-enumerated
        # y part up to the enumerated x part in the + operation.
        x_onehot = y.new_zeros(x.shape[:-1] + (self.args.hidden_dim,)).scatter_(-1, x, 1)
        y_conv = self.relu(self.conv(y.unsqueeze(-2))).reshape(y.shape[:-1] + (-1,))
        h = self.relu(self.x_to_hidden(x_onehot) + self.y_to_hidden(y_conv))
        return self.hidden_to_logits(h)


# We will create a single global instance later.
tones_generator = None


# The neural HMM model now uses tones_generator at each time step.
def model_5(sequences, lengths, args, batch_size=None, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    assert lengths.shape == (num_sequences,)
    assert lengths.max() <= max_length

    # Initialize a global module instance if needed.
    global tones_generator
    if tones_generator is None:
        tones_generator = TonesGenerator(args, data_dim)
    pyro.module("tones_generator", tones_generator)

    with poutine.mask(mask=include_prior):
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                  .independent(1))
    with pyro.plate("sequences", len(sequences), batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        y = torch.zeros(data_dim)
        for t in pyro.markov(range(lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                # Note that since each tone depends on all tones at a previous time step
                # the tones at different time steps now need to live in separate plates.
                with pyro.plate("tones_{}".format(t), data_dim, dim=-1):
                    y = pyro.sample("y_{}".format(t),
                                    dist.Bernoulli(logits=tones_generator(x, y)),
                                    obs=sequences[batch, t])


models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}


def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')
    data = poly.load_data()

    logging.info('-' * 40)
    model = models[args.model]
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(data['train']['sequences'])))
    sequences = torch.tensor(data['train']['sequences'], dtype=torch.float32)
    lengths = torch.tensor(data['train']['sequence_lengths'], dtype=torch.long)
    if args.truncate:
        lengths.clamp_(max=args.truncate)
    num_observations = float(lengths.sum())
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=2)
    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)

    # We'll train on small minibatches.
    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(sequences, lengths, args=args, batch_size=args.batch_size)
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, sequences, lengths, args, include_prior=False)
    logging.info('training loss = {}'.format(train_loss / num_observations))

    # Finally we evaluate on the test dataset.
    logging.info('-' * 40)
    logging.info('Evaluating on {} test sequences'.format(len(data['test']['sequences'])))
    sequences = torch.tensor(data['test']['sequences'], dtype=torch.float32)
    lengths = torch.tensor(data['test']['sequence_lengths'], dtype=torch.long)
    if args.truncate:
        lengths.clamp_(max=args.truncate)
    num_observations = float(lengths.sum())
    test_loss = elbo.loss(model, guide, sequences, lengths, args=args, include_prior=False)
    logging.info('test loss = {}'.format(test_loss / num_observations))

    # We expect models with higher capacity to perform better,
    # but eventually overfit to the training set.
    capacity = sum(value.reshape(-1).size(0)
                   for value in pyro.get_param_store().values())
    logging.info('{} capacity = {} parameters'.format(model.__name__, capacity))


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.0')
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
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
