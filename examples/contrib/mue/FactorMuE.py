# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
A PCA model with a MuE emission (FactorMuE). Uses the MuE package.
"""

import torch
import torch.nn as nn
from torch.nn.functional import softplus

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

from pyro.contrib.mue.statearrangers import profile
from pyro.contrib.mue.variablelengthhmm import VariableLengthDiscreteHMM

import datetime
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, obs_seq_length, alphabet_length, z_dim):
        super().__init__()

        self.input_size = obs_seq_length * alphabet_length
        self.f1_mn = nn.Linear(self.input_size, z_dim)
        self.f1_sd = nn.Linear(self.input_size, z_dim)

    def forward(self, data):

        data = data.reshape(-1, self.input_size)
        z_loc = self.f1_mn(data)
        z_scale = softplus(self.f1_sd(data))

        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, obs_seq_length, alphabet_length, z_dim):
        super().__init__()

        self.obs_seq_length = obs_seq_length
        self.alphabet_length = alphabet_length
        self.output_size = 2 * obs_seq_length * alphabet_length
        self.f = nn.Linear(z_dim, self.output_size)

    def forward(self, z):

        seq = self.f(z)
        seq = seq.reshape([-1, 2, self.obs_seq_length,
                           self.alphabet_length])
        return seq


class FactorMuE(nn.Module):

    def __init__(self, obs_seq_length, alphabet_length, z_dim,
                 latent_seq_length=None, prior_scale=1.,
                 indel_prior_strength=10.):
        super().__init__()

        # Constants.
        assert isinstance(obs_seq_length, int) and obs_seq_length > 0
        self.obs_seq_length = obs_seq_length
        if latent_seq_length is None:
            latent_seq_length = obs_seq_length
        else:
            assert isinstance(latent_seq_length, int) and latent_seq_length > 0
        assert isinstance(alphabet_length, int) and alphabet_length > 0
        self.alphabet_length = alphabet_length
        assert isinstance(z_dim, int) and z_dim > 0
        self.z_dim = z_dim

        # Parameter shapes.
        self.seq_shape = (latent_seq_length+1, alphabet_length)
        self.indel_shape = (latent_seq_length+1, 3, 2)

        assert isinstance(prior_scale, float)
        self.prior_scale = prior_scale
        assert isinstance(indel_prior_strength, float)
        self.indel_prior = torch.tensor([indel_prior_strength, 0.])

        # Initialize layers.
        self.encoder = Encoder(obs_seq_length, alphabet_length, z_dim)
        self.decoder = Decoder(obs_seq_length, alphabet_length, z_dim)
        self.statearrange = profile(latent_seq_length)

    def model(self, data):

        pyro.module("decoder", self.decoder)

        # Indel probabilities.
        insert = pyro.sample("insert", dist.Normal(
                self.indel_prior * torch.ones(self.indel_shape),
                self.prior_scale * torch.ones(self.indel_shape)).to_event(3))
        insert_logits = insert - insert.logsumexp(-1, True)
        delete = pyro.sample("delete", dist.Normal(
                self.indel_prior * torch.ones(self.indel_shape),
                self.prior_scale * torch.ones(self.indel_shape)).to_event(3))
        delete_logits = delete - delete.logsumexp(-1, True)

        # Temperature.
        # pyro.sample("inverse_temp", dist.Normal())

        with pyro.plate("batch", data.shape[0]), poutine.scale(
                    scale=self.scale_factor):
            # Sample latent variable from prior.
            z = pyro.sample("latent", dist.Normal(
                torch.zeros(self.z_dim), torch.ones(self.z_dim)).to_event(1))
            # Decode latent sequence.
            latent_seq = self.decoder.forward(z)
            # Construct ancestral and insertion sequences.
            ancestor_seq_logits = latent_seq[..., 0, :, :]
            ancestor_seq_logits = (ancestor_seq_logits -
                                   ancestor_seq_logits.logsumexp(-1, True))
            insert_seq_logits = latent_seq[..., 1, :, :]
            insert_seq_logits = (insert_seq_logits -
                                 insert_seq_logits.logsumexp(-1, True))
            # Construct HMM parameters.
            initial_logits, transition_logits, observation_logits = (
                    self.statearrange(ancestor_seq_logits, insert_seq_logits,
                                      insert_logits, delete_logits))
            # Draw samples.
            pyro.sample("obs",
                        VariableLengthDiscreteHMM(initial_logits,
                                                  transition_logits,
                                                  observation_logits),
                        obs=data)
