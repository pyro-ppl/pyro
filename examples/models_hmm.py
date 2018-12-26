from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
import torch.nn as nn
from torch.distributions import constraints

import dmm.polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam, ClippedAdam


# HMM
class model1(nn.Module):
    def __init__(self, args, data_dim):
        super(model1, self).__init__()

    def model(self, sequences, lengths, args, mb=None, scale=1.0):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args['hidden_dim']
        probs_x = pyro.param("probs_x", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_y = pyro.param("probs_y", lambda: torch.rand(hidden_dim, data_dim),
                             constraint=constraints.unit_interval)
        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", mb.size(0), subsample=mb, dim=-2) as batch:
            lengths = lengths[batch]
            with pyro.poutine.scale(scale=scale):
                x = 0
                for t in pyro.markov(range(lengths.max())):
                    with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                        x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                        infer={"enumerate": "parallel"})
                        with tones_plate:
                            pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[x.squeeze(-1)]),
                                        obs=sequences[batch, t])

# IFHMM
# formerly called model3
class model2(nn.Module):
    def __init__(self, args, data_dim):
        super(model2, self).__init__()

    def model(self, sequences, lengths, args, mb=None, scale=1.0):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args['hidden_dim']
        probs_w = pyro.param("probs_w", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_x = pyro.param("probs_x", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_y = pyro.param("probs_y", lambda: torch.rand(hidden_dim, hidden_dim, data_dim),
                             constraint=constraints.unit_interval)
        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", mb.size(0), subsample=mb, dim=-2) as batch:
            lengths = lengths[batch]
            with pyro.poutine.scale(scale=scale):
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


# FHMM
# formerly called model4
class model3(nn.Module):
    def __init__(self, args, data_dim):
        super(model3, self).__init__()

    def model(self, sequences, lengths, args, mb=None, scale=1.0):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args['hidden_dim']
        hidden = torch.arange(hidden_dim, dtype=torch.long)
        probs_w = pyro.param("probs_w", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_x = pyro.param("probs_x", lambda: (0.9 * torch.eye(hidden_dim) + \
                                                 0.1 * torch.ones(hidden_dim, hidden_dim)).
                                                 unsqueeze(-1).expand(hidden_dim, hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_y = pyro.param("probs_y", lambda: torch.rand(hidden_dim, hidden_dim, data_dim),
                             constraint=constraints.unit_interval)
        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", mb.size(0), subsample=mb, dim=-2) as batch:
            lengths = lengths[batch]
            with pyro.poutine.scale(scale=scale):
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


# arHMM
# formerly called model2
class model4(nn.Module):
    def __init__(self, args, data_dim):
        super(model4, self).__init__()

    def model(self, sequences, lengths, args, mb=None, scale=1.0):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args['hidden_dim']
        probs_x = pyro.param("probs_x", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_y = pyro.param("probs_y", lambda: torch.rand(hidden_dim, 2, data_dim),
                             constraint=constraints.unit_interval)
        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", mb.size(0), subsample=mb, dim=-2) as batch:
            lengths = lengths[batch]
            with pyro.poutine.scale(scale=scale):
                x, y = 0, 0
                for t in pyro.markov(range(lengths.max())):
                    with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                        x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                        infer={"enumerate": "parallel"})
                        with tones_plate as tones:
                            y = pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[x, y, tones]),
                                            obs=sequences[batch, t]).long()


# arIFHMM
class model5(nn.Module):
    def __init__(self, args, data_dim):
        super(model5, self).__init__()

    def model(self, sequences, lengths, args, mb=None, scale=1.0):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args['hidden_dim']
        probs_w = pyro.param("probs_w", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_x = pyro.param("probs_x", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_y = pyro.param("probs_y", lambda: torch.rand(hidden_dim, hidden_dim, 2, data_dim),
                             constraint=constraints.unit_interval)
        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", mb.size(0), subsample=mb, dim=-2) as batch:
            lengths = lengths[batch]
            with pyro.poutine.scale(scale=scale):
                w, x, y = 0, 0, 0
                for t in pyro.markov(range(lengths.max())):
                    with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                        w = pyro.sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                        infer={"enumerate": "parallel"})
                        x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                        infer={"enumerate": "parallel"})
                        with tones_plate as tones:
                            y = pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[w, x, y, tones]),
                                            obs=sequences[batch, t]).long()


# arFHMM
class model6(nn.Module):
    def __init__(self, args, data_dim):
        super(model6, self).__init__()

    def model(self, sequences, lengths, args, mb=None, scale=1.0):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args['hidden_dim']
        hidden = torch.arange(hidden_dim, dtype=torch.long)
        probs_w = pyro.param("probs_w", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_x = pyro.param("probs_x", lambda: (0.9 * torch.eye(hidden_dim) + \
                                                 0.1 * torch.ones(hidden_dim, hidden_dim)).
                                                 unsqueeze(-1).expand(hidden_dim, hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_y = pyro.param("probs_y", lambda: torch.rand(hidden_dim, hidden_dim, 2, data_dim),
                             constraint=constraints.unit_interval)
        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", mb.size(0), subsample=mb, dim=-2) as batch:
            lengths = lengths[batch]
            with pyro.poutine.scale(scale=scale):
                w = x = y = torch.tensor(0, dtype=torch.long)
                for t in pyro.markov(range(lengths.max())):
                    with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                        w = pyro.sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                        infer={"enumerate": "parallel"})
                        x = pyro.sample("x_{}".format(t),
                                        dist.Categorical(probs_x[w.unsqueeze(-1), x.unsqueeze(-1), hidden]),
                                        infer={"enumerate": "parallel"})
                        with tones_plate as tones:
                            pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[w, x, y, tones]),
                                        obs=sequences[batch, t])

# takes x and y as input
class TonesGenerator(nn.Module):
    def __init__(self, args, data_dim):
        self.args = args
        self.data_dim = data_dim
        super(TonesGenerator, self).__init__()
        self.x_to_hidden = nn.Linear(args['hidden_dim'], args['nn_dim'])
        self.y_to_hidden = nn.Linear(args['nn_channels'] * data_dim, args['nn_dim'])
        self.conv = nn.Conv1d(1, args['nn_channels'], 3, padding=1)
        self.hidden_to_logits = nn.Linear(args['nn_dim'], data_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        if y.dim() < 2:
            y = y.unsqueeze(0)
        x_onehot = y.new_zeros(x.shape[:-1] + (self.args['hidden_dim'],)).scatter_(-1, x, 1)
        y_conv = self.relu(self.conv(y.unsqueeze(-2))).reshape(y.shape[:-1] + (-1,))
        h = self.relu(self.x_to_hidden(x_onehot) + self.y_to_hidden(y_conv))
        return self.hidden_to_logits(h)


# takes w, x and y as input
class TonesGenerator2(nn.Module):
    def __init__(self, args, data_dim):
        self.args = args
        self.data_dim = data_dim
        super(TonesGenerator2, self).__init__()
        self.x_to_hidden = nn.Linear(args['hidden_dim'], args['nn_dim'])
        self.w_to_hidden = nn.Linear(args['hidden_dim'], args['nn_dim'])
        self.y_to_hidden = nn.Linear(args['nn_channels'] * data_dim, args['nn_dim'])
        self.conv = nn.Conv1d(1, args['nn_channels'], 3, padding=1)
        self.hidden_to_logits = nn.Linear(args['nn_dim'], data_dim)
        self.relu = nn.ReLU()

    def forward(self, w, x, y):
        if y.dim() < 2:
            y = y.unsqueeze(0)
        w_onehot = y.new_zeros(w.shape[:-1] + (self.args['hidden_dim'],)).scatter_(-1, w, 1)
        x_onehot = y.new_zeros(x.shape[:-1] + (self.args['hidden_dim'],)).scatter_(-1, x, 1)
        y_conv = self.relu(self.conv(y.unsqueeze(-2))).reshape(y.shape[:-1] + (-1,))
        h = self.relu(self.x_to_hidden(x_onehot) + self.w_to_hidden(w_onehot) + self.y_to_hidden(y_conv))
        return self.hidden_to_logits(h)


# nnHMM
# formerly called model5
class model7(nn.Module):
    def __init__(self, args, data_dim):
        super(model7, self).__init__()
        self.tones_generator = TonesGenerator(args, data_dim)

    def model(self, sequences, lengths, args, mb=None, scale=1.0):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args['hidden_dim']
        pyro.module("tones_generator", self.tones_generator)
        probs_x = pyro.param("probs_x", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        with pyro.plate("sequences", mb.size(0), subsample=mb, dim=-2) as batch:
            lengths = lengths[batch]
            with pyro.poutine.scale(scale=scale):
                x = 0
                y = torch.zeros(data_dim)
                for t in pyro.markov(range(lengths.max())):
                    with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                        x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                        infer={"enumerate": "parallel"})
                        with pyro.plate("tones_{}".format(t), data_dim, dim=-1):
                            y = pyro.sample("y_{}".format(t),
                                            dist.Bernoulli(logits=self.tones_generator(x, y)),
                                            obs=sequences[batch, t])


# nnIFHMM
class model8(nn.Module):
    def __init__(self, args, data_dim):
        super(model8, self).__init__()
        self.tones_generator = TonesGenerator2(args, data_dim)

    def model(self, sequences, lengths, args, mb=None, scale=1.0):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args['hidden_dim']
        pyro.module("tones_generator", self.tones_generator)
        probs_w = pyro.param("probs_w", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_x = pyro.param("probs_x", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        with pyro.plate("sequences", mb.size(0), subsample=mb, dim=-2) as batch:
            lengths = lengths[batch]
            with pyro.poutine.scale(scale=scale):
                w, x = 0, 0
                y = torch.zeros(data_dim)
                for t in pyro.markov(range(lengths.max())):
                    with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                        w = pyro.sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                        infer={"enumerate": "parallel"})
                        x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                        infer={"enumerate": "parallel"})
                        with pyro.plate("tones_{}".format(t), data_dim, dim=-1):
                            y = pyro.sample("y_{}".format(t),
                                            dist.Bernoulli(logits=self.tones_generator(w, x, y)),
                                            obs=sequences[batch, t])


# nnIFHMM
class model9(nn.Module):
    def __init__(self, args, data_dim):
        super(model9, self).__init__()
        self.tones_generator = TonesGenerator2(args, data_dim)

    def model(self, sequences, lengths, args, mb=None, scale=1.0):
        num_sequences, max_length, data_dim = sequences.shape
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        hidden_dim = args['hidden_dim']
        pyro.module("tones_generator", self.tones_generator)
        hidden = torch.arange(hidden_dim, dtype=torch.long)
        probs_w = pyro.param("probs_w", lambda: 0.9 * torch.eye(hidden_dim) + \
                                                0.1 * torch.ones(hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_x = pyro.param("probs_x", lambda: (0.9 * torch.eye(hidden_dim) + \
                                                 0.1 * torch.ones(hidden_dim, hidden_dim)).
                                                 unsqueeze(-1).expand(hidden_dim, hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
        probs_y = pyro.param("probs_y", lambda: torch.rand(hidden_dim, hidden_dim, 2, data_dim),
                             constraint=constraints.unit_interval)
        tones_plate = pyro.plate("tones", data_dim, dim=-1)
        with pyro.plate("sequences", mb.size(0), subsample=mb, dim=-2) as batch:
            lengths = lengths[batch]
            with pyro.poutine.scale(scale=scale):
                w = x = torch.tensor(0, dtype=torch.long)
                y = torch.zeros(data_dim)
                for t in pyro.markov(range(lengths.max())):
                    with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                        w = pyro.sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                        infer={"enumerate": "parallel"})
                        x = pyro.sample("x_{}".format(t),
                                        dist.Categorical(probs_x[w.unsqueeze(-1), x.unsqueeze(-1), hidden]),
                                        infer={"enumerate": "parallel"})
                        with pyro.plate("tones_{}".format(t), data_dim, dim=-1):
                            y = pyro.sample("y_{}".format(t),
                                            dist.Bernoulli(logits=self.tones_generator(w, x, y)),
                                            obs=sequences[batch, t])
