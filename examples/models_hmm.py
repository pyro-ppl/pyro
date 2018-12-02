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
from pyro.optim import Adam, ClippedAdam


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

class model1(nn.Module):
    def __init__(self, args, data_dim):
        super(model1, self).__init__()

    def model(self, sequences, lengths, args, mb=None, include_prior=True):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        with poutine.mask(mask=torch.tensor(include_prior)):
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
        with pyro.plate("sequences", len(sequences), subsample=mb, dim=-2) as batch:
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
class model2(nn.Module):
    def __init__(self, args, data_dim):
        super(model2, self).__init__()

    def model(self, sequences, lengths, args, mb=None, include_prior=True):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        with poutine.mask(mask=torch.tensor(include_prior)):
            probs_x = pyro.sample("probs_x",
                                  dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                      .independent(1))
            probs_y = pyro.sample("probs_y",
                                  dist.Beta(0.1, 0.9)
                                      .expand([args.hidden_dim, 2, data_dim])
                                      .independent(3))
        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", len(sequences), subsample=mb, dim=-2) as batch:
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
class model3(nn.Module):
    def __init__(self, args, data_dim):
        super(model3, self).__init__()

    def model(self, sequences, lengths, args, mb=None, include_prior=True):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
        with poutine.mask(mask=torch.tensor(include_prior)):
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
        with pyro.plate("sequences", len(sequences), subsample=mb, dim=-2) as batch:
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
class model4(nn.Module):
    def __init__(self, args, data_dim):
        super(model4, self).__init__()

    def model(self, sequences, lengths, args, mb=None, include_prior=True):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
        hidden = torch.arange(hidden_dim, dtype=torch.long)
        with poutine.mask(mask=torch.tensor(include_prior)):
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
        with pyro.plate("sequences", len(sequences), subsample=mb, dim=-2) as batch:
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
        if y.dim() < 2:
            y = y.unsqueeze(0)
        x_onehot = y.new_zeros(x.shape[:-1] + (self.args.hidden_dim,)).scatter_(-1, x, 1)
        y_conv = self.relu(self.conv(y.unsqueeze(-2))).reshape(y.shape[:-1] + (-1,))
        h = self.relu(self.x_to_hidden(x_onehot) + self.y_to_hidden(y_conv))
        return self.hidden_to_logits(h)



# The neural HMM model now uses tones_generator at each time step.
class model5(nn.Module):
    def __init__(self, args, data_dim):
        super(model5, self).__init__()
        self.tones_generator = TonesGenerator(args, data_dim)

    def model(self, sequences, lengths, args, mb=None, include_prior=True):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        pyro.module("tones_generator", self.tones_generator)

        with poutine.mask(mask=torch.tensor(include_prior)):
            probs_x = pyro.sample("probs_x",
                                  dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                      .independent(1))
        with pyro.plate("sequences", len(sequences), subsample=mb, dim=-2) as batch:
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
                                        dist.Bernoulli(logits=self.tones_generator(x, y)),
                                        obs=sequences[batch, t])
