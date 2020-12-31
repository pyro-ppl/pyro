# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
A standard profile HMM model example, using the MuE package.
"""

import torch
import torch.nn as nn
from torch.nn.functional import softplus

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

from pyro.contrib.mue.statearrangers import profile
from pyro.contrib.mue.variablelengthhmm import VariableLengthDiscreteHMM

import argparse
import datetime
import matplotlib.pyplot as plt


class ProfileHMM(nn.Module):

    def __init__(self, latent_seq_length, alphabet_length,
                 prior_scale=1., indel_prior_strength=10.):
        super().__init__()

        assert isinstance(latent_seq_length, int) and latent_seq_length > 0
        self.latent_seq_length = latent_seq_length
        assert isinstance(alphabet_length, int) and alphabet_length > 0
        self.alphabet_length = alphabet_length

        self.seq_shape = (latent_seq_length+1, alphabet_length)
        self.indel_shape = (latent_seq_length+1, 3, 2)

        assert isinstance(prior_scale, float)
        self.prior_scale = prior_scale
        assert isinstance(indel_prior_strength, float)
        self.indel_prior = torch.tensor([indel_prior_strength, 0.])

        # Initialize state arranger.
        self.statearrange = profile(latent_seq_length)

    def model(self, data):

        # Latent sequence.
        ancestor_seq = pyro.sample("ancestor_seq", dist.Normal(
                torch.zeros(self.seq_shape),
                self.prior_scale * torch.ones(self.seq_shape)).to_event(2))
        ancestor_seq_logits = ancestor_seq - ancestor_seq.logsumexp(-1, True)
        insert_seq = pyro.sample("insert_seq", dist.Normal(
                torch.zeros(self.seq_shape),
                self.prior_scale * torch.ones(self.seq_shape)).to_event(2))
        insert_seq_logits = insert_seq - insert_seq.logsumexp(-1, True)

        # Indel probabilities.
        insert = pyro.sample("insert", dist.Normal(
                self.indel_prior * torch.ones(self.indel_shape),
                self.prior_scale * torch.ones(self.indel_shape)).to_event(3))
        insert_logits = insert - insert.logsumexp(-1, True)
        delete = pyro.sample("delete", dist.Normal(
                self.indel_prior * torch.ones(self.indel_shape),
                self.prior_scale * torch.ones(self.indel_shape)).to_event(3))
        delete_logits = delete - delete.logsumexp(-1, True)

        # Construct HMM parameters.
        initial_logits, transition_logits, observation_logits = (
                self.statearrange(ancestor_seq_logits, insert_seq_logits,
                                  insert_logits, delete_logits))
        # Draw samples.
        for i in pyro.plate("data", data.shape[0]):
            pyro.sample("obs_{}".format(i),
                        VariableLengthDiscreteHMM(initial_logits,
                                                  transition_logits,
                                                  observation_logits),
                        obs=data[i])

    def guide(self, data):
        # Sequence.
        ancestor_seq_q_mn = pyro.param("ancestor_seq_q_mn",
                                       torch.zeros(self.seq_shape))
        ancestor_seq_q_sd = pyro.param("ancestor_seq_q_sd",
                                       torch.zeros(self.seq_shape))
        pyro.sample("ancestor_seq", dist.Normal(
                ancestor_seq_q_mn, softplus(ancestor_seq_q_sd)).to_event(2))
        insert_seq_q_mn = pyro.param("insert_seq_q_mn",
                                     torch.zeros(self.seq_shape))
        insert_seq_q_sd = pyro.param("insert_seq_q_sd",
                                     torch.zeros(self.seq_shape))
        pyro.sample("insert_seq", dist.Normal(
                insert_seq_q_mn, softplus(insert_seq_q_sd)).to_event(2))

        # Indels.
        insert_q_mn = pyro.param("insert_q_mn",
                                 torch.ones(self.indel_shape)
                                 * self.indel_prior)
        insert_q_sd = pyro.param("insert_q_sd",
                                 torch.zeros(self.indel_shape))
        pyro.sample("insert", dist.Normal(
                insert_q_mn, softplus(insert_q_sd)).to_event(3))
        delete_q_mn = pyro.param("delete_q_mn",
                                 torch.ones(self.indel_shape)
                                 * self.indel_prior)
        delete_q_sd = pyro.param("delete_q_sd",
                                 torch.zeros(self.indel_shape))
        pyro.sample("delete", dist.Normal(
                delete_q_mn, softplus(delete_q_sd)).to_event(3))


def main(args):

    torch.manual_seed(0)
    torch.set_default_tensor_type('torch.DoubleTensor')

    small_test = args.test

    if small_test:
        mult_dat = 1
        mult_step = 1
    else:
        mult_dat = 10
        mult_step = 10

    data = torch.cat([torch.tensor([[0., 1.],
                                    [1., 0.],
                                    [0., 1.],
                                    [0., 1.],
                                    [1., 0.],
                                    [0., 0.]])[None, :, :]
                      for j in range(6*mult_dat)] +
                     [torch.tensor([[0., 1.],
                                    [1., 0.],
                                    [1., 0.],
                                    [0., 1.],
                                    [0., 0.],
                                    [0., 0.]])[None, :, :]
                     for j in range(4*mult_dat)], dim=0)
    # Set up inference.
    latent_seq_length, alphabet_length = 6, 2
    adam_params = {"lr": 0.05, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)
    model = ProfileHMM(latent_seq_length, alphabet_length)

    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
    n_steps = 10*mult_step

    # Run inference.
    losses = []
    t0 = datetime.datetime.now()
    for step in range(n_steps):
        loss = svi.step(data)
        losses.append(loss)
        if step % 10 == 0:
            print(loss, ' ', datetime.datetime.now() - t0)

    # Plots.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(6, 6))
    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig('phmm.loss_{}.pdf'.format(time_stamp))

    plt.figure(figsize=(6, 6))
    ancestor_seq = pyro.param("ancestor_seq_q_mn").detach()
    ancestor_seq_expect = torch.exp(ancestor_seq -
                                    ancestor_seq.logsumexp(-1, True))
    plt.plot(ancestor_seq_expect[:, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of character 1')
    plt.savefig('phmm.ancestor_seq_prob_{}.pdf'.format(time_stamp))

    plt.figure(figsize=(6, 6))
    insert = pyro.param("insert_q_mn").detach()
    insert_expect = torch.exp(insert - insert.logsumexp(-1, True))
    plt.plot(insert_expect[:, :, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of insert')
    plt.savefig('phmm.insert_prob_{}.pdf'.format(time_stamp))
    plt.figure(figsize=(6, 6))
    delete = pyro.param("delete_q_mn").detach()
    delete_expect = torch.exp(delete - delete.logsumexp(-1, True))
    plt.plot(delete_expect[:, :, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of delete')
    plt.savefig('phmm.delete_prob_{}.pdf'.format(time_stamp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Basic profile HMM model (constant + MuE).")
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='small dataset, a few steps')
    args = parser.parse_args()
    main(args)
