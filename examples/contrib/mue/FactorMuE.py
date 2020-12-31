# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
A PCA model with a MuE emission (FactorMuE). Uses the MuE package.
"""

import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.optim import Adam

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.mue.statearrangers import profile
from pyro.contrib.mue.variablelengthhmm import VariableLengthDiscreteHMM
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import MultiStepLR


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
    def __init__(self, latent_seq_length, alphabet_length, z_dim):
        super().__init__()

        self.latent_seq_length = latent_seq_length
        self.alphabet_length = alphabet_length
        self.output_size = 2 * (latent_seq_length+1) * alphabet_length
        self.f = nn.Linear(z_dim, self.output_size)

    def forward(self, z):

        seq = self.f(z)
        seq = seq.reshape([-1, 2, self.latent_seq_length+1,
                           self.alphabet_length])
        return seq


class FactorMuE(nn.Module):

    def __init__(self, obs_seq_length, alphabet_length, z_dim,
                 scale_factor=1.,
                 latent_seq_length=None, prior_scale=1.,
                 indel_prior_strength=10., inverse_temp_prior=100.):
        super().__init__()

        # Constants.
        assert isinstance(obs_seq_length, int) and obs_seq_length > 0
        self.obs_seq_length = obs_seq_length
        if latent_seq_length is None:
            latent_seq_length = obs_seq_length
        else:
            assert isinstance(latent_seq_length, int) and latent_seq_length > 0
        self.latent_seq_length = latent_seq_length
        assert isinstance(alphabet_length, int) and alphabet_length > 0
        self.alphabet_length = alphabet_length
        assert isinstance(z_dim, int) and z_dim > 0
        self.z_dim = z_dim

        # Parameter shapes.
        self.seq_shape = (latent_seq_length+1, alphabet_length)
        self.indel_shape = (latent_seq_length+1, 3, 2)

        # Priors.
        assert isinstance(prior_scale, float)
        self.prior_scale = torch.tensor(prior_scale)
        assert isinstance(indel_prior_strength, float)
        self.indel_prior = torch.tensor([indel_prior_strength, 0.])
        assert isinstance(inverse_temp_prior, float)
        self.inverse_temp_prior = torch.tensor(inverse_temp_prior)

        # Batch control.
        self.scale_factor = scale_factor

        # Initialize layers.
        self.encoder = Encoder(obs_seq_length, alphabet_length, z_dim)
        self.decoder = Decoder(latent_seq_length, alphabet_length, z_dim)
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

        # Inverse temperature.
        inverse_temp = pyro.sample("inverse_temp", dist.Normal(
                self.inverse_temp_prior, torch.tensor(1.)))

        with pyro.plate("batch", data.shape[0]), poutine.scale(
                    scale=self.scale_factor):
            # Sample latent variable from prior.
            z = pyro.sample("latent", dist.Normal(
                torch.zeros(self.z_dim), torch.ones(self.z_dim)).to_event(1))
            # Decode latent sequence.
            latent_seq = self.decoder(z)
            # Construct ancestral and insertion sequences.
            ancestor_seq_logits = (latent_seq[..., 0, :, :] *
                                   softplus(inverse_temp))
            ancestor_seq_logits = (ancestor_seq_logits -
                                   ancestor_seq_logits.logsumexp(-1, True))
            insert_seq_logits = (latent_seq[..., 1, :, :] *
                                 softplus(inverse_temp))
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

    def guide(self, data):
        # Register encoder with pyro.
        pyro.module("encoder", self.encoder)

        # Indel probabilities.
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
        # Inverse temperature.
        inverse_temp_q_mn = pyro.param("inverse_temp_q_mn", torch.tensor(0.))
        inverse_temp_q_sd = pyro.param("inverse_temp_q_sd", torch.tensor(0.))
        pyro.sample("inverse_temp", dist.Normal(
                inverse_temp_q_mn, softplus(inverse_temp_q_sd)))

        # Per data latent variables.
        with pyro.plate("batch", data.shape[0]), poutine.scale(
                    scale=self.scale_factor):
            # Encode seq.
            z_loc, z_scale = self.encoder(data)
            # Sample.
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_ancestor_seq(self, data, inverse_temp=1.):
        # Encode seq.
        z_loc = self.encoder(data)[0]
        # Reconstruct
        latent_seq = self.decoder(z_loc)
        # Construct ancestral sequence.
        ancestor_seq_logits = latent_seq[..., 0, :, :] * softplus(inverse_temp)
        ancestor_seq_logits = (ancestor_seq_logits -
                               ancestor_seq_logits.logsumexp(-1, True))
        return torch.exp(ancestor_seq_logits)


def main():

    torch.manual_seed(9)
    torch.set_default_tensor_type('torch.DoubleTensor')

    small_test = False

    if small_test:
        mult_dat = 1
        mult_step = 1
    else:
        mult_dat = 10
        mult_step = 400

    # Construct example dataset.
    xs = [torch.tensor([[0., 1.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.],
                        [1., 0.],
                        [0., 0.]]),
          torch.tensor([[0., 1.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 0.],
                        [0., 0.]]),
          torch.tensor([[0., 1.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 0.]])]
    data = torch.cat([xs[0][None, :, :] for j in range(6*mult_dat)] +
                     [xs[1][None, :, :] for j in range(4*mult_dat)] +
                     [xs[2][None, :, :] for j in range(4*mult_dat)], dim=0)
    # Set up inference.
    obs_seq_length, alphabet_length, z_dim = 6, 2, 2
    # adam_params = {"lr": 0.1, "betas": (0.90, 0.999)}
    scheduler = MultiStepLR({'optimizer': Adam,
                             'optim_args': {'lr': 0.1},
                             'milestones': [20, 100, 1000, 2000],
                             'gamma': 0.5})
    # optimizer = Adam(adam_params)
    model = FactorMuE(obs_seq_length, alphabet_length, z_dim)

    svi = SVI(model.model, model.guide, scheduler, loss=Trace_ELBO())
    n_steps = 10*mult_step

    # Run inference.
    losses = []
    t0 = datetime.datetime.now()
    for step in range(n_steps):

        loss = svi.step(data)
        losses.append(loss)
        scheduler.step()
        if step % 10 == 0:
            print(step, loss, ' ', datetime.datetime.now() - t0)

    # Plots.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(6, 6))
    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig('FactorMuE/loss_{}.pdf'.format(time_stamp))

    plt.figure(figsize=(6, 6))
    latent = model.encoder(data)[0].detach()
    plt.scatter(latent[:, 0], latent[:, 1])
    plt.xlabel('z_1')
    plt.ylabel('z_2')
    plt.savefig('FactorMuE/latent_{}.pdf'.format(time_stamp))

    plt.figure(figsize=(6, 6))
    decoder_bias = pyro.param('decoder$$$f.bias').detach()
    decoder_bias = decoder_bias.reshape(
                    [-1, 2, model.latent_seq_length+1, model.alphabet_length])
    plt.plot(decoder_bias[0, 0, :, 1])
    plt.xlabel('position')
    plt.ylabel('bias for character 1')
    plt.savefig('FactorMuE/decoder_bias_{}.pdf'.format(time_stamp))

    for xi, x in enumerate(xs):
        reconstruct_x = model.reconstruct_ancestor_seq(
                x, pyro.param("inverse_temp_q_mn")).detach()
        plt.figure(figsize=(6, 6))
        plt.plot(reconstruct_x[0, :, 1], label="reconstruct")
        plt.plot(x[:, 1], label="data")
        plt.xlabel('position')
        plt.ylabel('probability of character 1')
        plt.legend()
        plt.savefig('FactorMuE/reconstruction_{}_{}.pdf'.format(
                        xi, time_stamp))

    plt.figure(figsize=(6, 6))
    insert = pyro.param("insert_q_mn").detach()
    insert_expect = torch.exp(insert - insert.logsumexp(-1, True))
    plt.plot(insert_expect[:, :, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of insert')
    plt.savefig('FactorMuE/insert_prob_{}.pdf'.format(time_stamp))
    plt.figure(figsize=(6, 6))
    delete = pyro.param("delete_q_mn").detach()
    delete_expect = torch.exp(delete - delete.logsumexp(-1, True))
    plt.plot(delete_expect[:, :, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of delete')
    plt.savefig('FactorMuE/delete_prob_{}.pdf'.format(time_stamp))


if __name__ == '__main__':
    main()
