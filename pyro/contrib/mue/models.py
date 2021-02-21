# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example MuE observation models.
"""

import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.optim import Adam
from torch.utils.data import DataLoader

import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.contrib.mue.missingdatahmm import MissingDataDiscreteHMM
from pyro.contrib.mue.statearrangers import Profile
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import MultiStepLR

import pdb


class ProfileHMM(nn.Module):
    """Model: Constant + MuE. """
    def __init__(self, latent_seq_length, alphabet_length,
                 length_model=False, prior_scale=1., indel_prior_bias=10.):
        super().__init__()

        assert isinstance(latent_seq_length, int) and latent_seq_length > 0
        self.latent_seq_length = latent_seq_length
        assert isinstance(alphabet_length, int) and alphabet_length > 0
        self.alphabet_length = alphabet_length

        self.precursor_seq_shape = (latent_seq_length, alphabet_length)
        self.insert_seq_shape = (latent_seq_length+1, alphabet_length)
        self.indel_shape = (latent_seq_length, 3, 2)

        assert isinstance(length_model, bool)
        self.length_model = length_model
        assert isinstance(prior_scale, float)
        self.prior_scale = prior_scale
        assert isinstance(indel_prior_bias, float)
        self.indel_prior = torch.tensor([indel_prior_bias, 0.])

        # Initialize state arranger.
        self.statearrange = Profile(latent_seq_length)

    def model(self, seq_data, L_data, local_scale):

        # Latent sequence.
        precursor_seq = pyro.sample("precursor_seq", dist.Normal(
                torch.zeros(self.precursor_seq_shape),
                self.prior_scale *
                torch.ones(self.precursor_seq_shape)).to_event(2))
        precursor_seq_logits = precursor_seq - precursor_seq.logsumexp(-1, True)
        insert_seq = pyro.sample("insert_seq", dist.Normal(
                torch.zeros(self.insert_seq_shape),
                self.prior_scale *
                torch.ones(self.insert_seq_shape)).to_event(2))
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
                self.statearrange(precursor_seq_logits, insert_seq_logits,
                                  insert_logits, delete_logits))

        # Length model.
        if self.length_model:
            length = pyro.sample("length", dist.Normal(
                    torch.tensor(200.), torch.tensor(1000.)))
            L_mean = softplus(length)

        with pyro.plate("batch", L_data.shape[0]):
            with poutine.scale(scale=local_scale):

                if self.length_model:
                    pyro.sample("obs_L", dist.Poisson(L_mean),
                                obs=L_data)
                pyro.sample("obs_seq",
                            MissingDataDiscreteHMM(initial_logits,
                                                   transition_logits,
                                                   observation_logits),
                            obs=seq_data)

    def guide(self, seq_data, L_data, local_scale):
        # Sequence.
        precursor_seq_q_mn = pyro.param("precursor_seq_q_mn",
                                        torch.zeros(self.precursor_seq_shape))
        precursor_seq_q_sd = pyro.param("precursor_seq_q_sd",
                                        torch.zeros(self.precursor_seq_shape))
        pyro.sample("precursor_seq", dist.Normal(
                precursor_seq_q_mn, softplus(precursor_seq_q_sd)).to_event(2))
        insert_seq_q_mn = pyro.param("insert_seq_q_mn",
                                     torch.zeros(self.insert_seq_shape))
        insert_seq_q_sd = pyro.param("insert_seq_q_sd",
                                     torch.zeros(self.insert_seq_shape))
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

        # Length.
        if self.length_model:
            length_q_mn = pyro.param("length_q_mn", torch.zeros(1))
            length_q_sd = pyro.param("length_q_sd", torch.zeros(1))
            pyro.sample("length", dist.Normal(
                    length_q_mn, softplus(length_q_sd)))

    def fit_svi(self, dataset, epochs=1, batch_size=1, scheduler=None,
                jit=False):
        """Infer model parameters with stochastic variational inference."""

        # Setup.
        if batch_size is not None:
            self.batch_size = batch_size
        if scheduler is None:
            scheduler = MultiStepLR({'optimizer': Adam,
                                     'optim_args': {'lr': 0.01},
                                     'milestones': [],
                                     'gamma': 0.5})
        self.guide(None, None, None)
        if jit:
            Elbo = JitTrace_ELBO(ignore_jit_warnings=True)
        else:
            Elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, scheduler, loss=Elbo)
        dataload = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Run inference.
        losses = []
        t0 = datetime.datetime.now()
        for epoch in range(epochs):
            for seq_data, L_data in dataload:
                loss = svi.step(seq_data, L_data,
                                torch.tensor(dataset.data_size/L_data.shape[0]))
                losses.append(loss)
                scheduler.step()
            print(epoch, loss, ' ', datetime.datetime.now() - t0)
        return losses

    def evaluate(self, dataset_train, dataset_test, jit=False):
        """Evaluate performance on train and test datasets."""
        self.guide(None, None, None)
        if jit:
            Elbo = JitTrace_ELBO(ignore_jit_warnings=True)
        else:
            Elbo = Trace_ELBO()
        scheduler = MultiStepLR({'optimizer': Adam,
                                 'optim_args': {'lr': 0.01},
                                 'milestones': [],
                                 'gamma': 0.5})
        svi = SVI(self.model, self.guide, scheduler, loss=Elbo)
        dataload_train = DataLoader(dataset_train, batch_size=1, shuffle=False)
        dataload_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
        train_lp, train_perplex = 0., 0.
        for seq_data, L_data in dataload_train:
            lp = svi.evaluate_loss(
                seq_data, L_data, torch.tensor(dataset_train.data_size))
            train_lp += -lp
            train_perplex += lp / (L_data[0] + int(self.length_model))
        train_perplex = np.exp(train_perplex)
        test_lp, test_perplex = 0., 0.
        for seq_data, L_data in dataload_test:
            lp = svi.evaluate_loss(
                seq_data, L_data, torch.tensor(dataset_test.data_size))
            test_lp += -lp
            test_perplex += lp / (L_data[0].numpy() + int(self.length_model))
        test_perplex = np.exp(test_perplex)
        return train_lp, test_lp, train_perplex, test_perplex


class Encoder(nn.Module):
    def __init__(self, data_length, alphabet_length, z_dim):
        super().__init__()

        self.input_size = data_length * alphabet_length
        self.f1_mn = nn.Linear(self.input_size, z_dim)
        self.f1_sd = nn.Linear(self.input_size, z_dim)

    def forward(self, data):

        data = data.reshape(-1, self.input_size)
        z_loc = self.f1_mn(data)
        z_scale = softplus(self.f1_sd(data))

        return z_loc, z_scale


class FactorMuE(nn.Module):
    """Model: pPCA + MuE."""
    def __init__(self, data_length, alphabet_length, z_dim,
                 batch_size=10,
                 latent_seq_length=None,
                 indel_factor_dependence=False,
                 indel_prior_scale=1.,
                 indel_prior_bias=10.,
                 inverse_temp_prior=100.,
                 weights_prior_scale=1.,
                 offset_prior_scale=1.,
                 z_prior_distribution='Normal',
                 ARD_prior=False,
                 substitution_matrix=True,
                 substitution_prior_scale=10.,
                 latent_alphabet_length=None,
                 length_model=False,
                 epsilon=1e-32):
        super().__init__()

        # Constants.
        assert isinstance(data_length, int) and data_length > 0
        self.data_length = data_length
        if latent_seq_length is None:
            latent_seq_length = data_length
        else:
            assert isinstance(latent_seq_length, int) and latent_seq_length > 0
        self.latent_seq_length = latent_seq_length
        assert isinstance(alphabet_length, int) and alphabet_length > 0
        self.alphabet_length = alphabet_length
        assert isinstance(z_dim, int) and z_dim > 0
        self.z_dim = z_dim

        # Parameter shapes.
        if (not substitution_matrix) or (latent_alphabet_length is None):
            latent_alphabet_length = alphabet_length
        self.latent_alphabet_length = latent_alphabet_length
        self.indel_shape = (latent_seq_length, 3, 2)
        self.total_factor_size = (
                (2*latent_seq_length+1)*latent_alphabet_length +
                2*indel_factor_dependence*latent_seq_length*3*2 +
                length_model)

        # Architecture.
        self.indel_factor_dependence = indel_factor_dependence
        self.ARD_prior = ARD_prior
        self.substitution_matrix = substitution_matrix
        self.length_model = length_model

        # Priors.
        assert isinstance(indel_prior_scale, float)
        self.indel_prior_scale = torch.tensor(indel_prior_scale)
        assert isinstance(indel_prior_bias, float)
        self.indel_prior = torch.tensor([indel_prior_bias, 0.])
        assert isinstance(inverse_temp_prior, float)
        self.inverse_temp_prior = torch.tensor(inverse_temp_prior)
        assert isinstance(weights_prior_scale, float)
        self.weights_prior_scale = torch.tensor(weights_prior_scale)
        assert isinstance(offset_prior_scale, float)
        self.offset_prior_scale = torch.tensor(offset_prior_scale)
        assert isinstance(epsilon, float)
        self.epsilon = torch.tensor(epsilon)
        assert isinstance(substitution_prior_scale, float)
        self.substitution_prior_scale = torch.tensor(substitution_prior_scale)
        self.z_prior_distribution = z_prior_distribution

        # Batch control.
        assert isinstance(batch_size, int)
        self.batch_size = batch_size

        # Initialize layers.
        self.encoder = Encoder(data_length, alphabet_length, z_dim)
        self.statearrange = Profile(latent_seq_length)

    def decoder(self, z, W, B, inverse_temp):

        # Project.
        v = torch.mm(z, W) + B

        out = dict()
        if self.length_model:
            # Extract expected length.
            L_v = v[:, -1]
            out['L_mean'] = softplus(L_v)
        if self.indel_factor_dependence:
            # Extract insertion and deletion parameters.
            ind0 = (2*self.latent_seq_length+1)*self.latent_alphabet_length
            ind1 = ind0 + self.latent_seq_length*3*2
            ind2 = ind1 + self.latent_seq_length*3*2
            insert_v, delete_v = v[:, ind0:ind1], v[:, ind1:ind2]
            insert_v = (insert_v.reshape([-1, self.latent_seq_length, 3, 2])
                        + self.indel_prior)
            out['insert_logits'] = insert_v - insert_v.logsumexp(-1, True)
            delete_v = (delete_v.reshape([-1, self.latent_seq_length, 3, 2])
                        + self.indel_prior)
            out['delete_logits'] = delete_v - delete_v.logsumexp(-1, True)
        # Extract precursor and insertion sequences.
        ind0 = self.latent_seq_length*self.latent_alphabet_length
        ind1 = ind0 + (self.latent_seq_length+1)*self.latent_alphabet_length
        precursor_seq_v, insert_seq_v = v[:, :ind0], v[:, ind0:ind1]
        precursor_seq_v = (precursor_seq_v*softplus(inverse_temp)).reshape([
                -1, self.latent_seq_length, self.latent_alphabet_length])
        out['precursor_seq_logits'] = (
                precursor_seq_v - precursor_seq_v.logsumexp(-1, True))
        insert_seq_v = (insert_seq_v*softplus(inverse_temp)).reshape([
                -1, self.latent_seq_length+1, self.latent_alphabet_length])
        out['insert_seq_logits'] = (
                insert_seq_v - insert_seq_v.logsumexp(-1, True))

        return out

    def model(self, seq_data, L_data, local_scale, local_prior_scale):

        # ARD prior.
        if self.ARD_prior:
            # Relevance factors
            alpha = pyro.sample("alpha", dist.Gamma(
                    torch.ones(self.z_dim), torch.ones(self.z_dim)).to_event(1))
        else:
            alpha = torch.ones(self.z_dim)

        # Factor and offset.
        W = pyro.sample("W", dist.Normal(
                torch.zeros([self.z_dim, self.total_factor_size]),
                torch.ones([self.z_dim, self.total_factor_size]) *
                self.weights_prior_scale / (alpha[:, None] + self.epsilon)
                ).to_event(2))
        B = pyro.sample("B", dist.Normal(
                torch.zeros(self.total_factor_size),
                torch.ones(self.total_factor_size) * self.offset_prior_scale
                ).to_event(1))

        # Indel probabilities.
        if not self.indel_factor_dependence:
            insert = pyro.sample("insert", dist.Normal(
                self.indel_prior * torch.ones(self.indel_shape),
                self.indel_prior_scale * torch.ones(self.indel_shape)
                ).to_event(3))
            insert_logits = insert - insert.logsumexp(-1, True)
            delete = pyro.sample("delete", dist.Normal(
                self.indel_prior * torch.ones(self.indel_shape),
                self.indel_prior_scale * torch.ones(self.indel_shape)
                ).to_event(3))
            delete_logits = delete - delete.logsumexp(-1, True)

        # Inverse temperature.
        inverse_temp = pyro.sample("inverse_temp", dist.Normal(
                self.inverse_temp_prior, torch.tensor(1.)))

        # Substitution matrix.
        if self.substitution_matrix:
            substitute = pyro.sample("substitute", dist.Normal(
                torch.zeros([
                        self.latent_alphabet_length, self.alphabet_length]),
                self.substitution_prior_scale * torch.ones([
                        self.latent_alphabet_length, self.alphabet_length])
                ).to_event(2))

        with pyro.plate("batch", L_data.shape[0]):
            with poutine.scale(scale=local_scale):
                with poutine.scale(scale=local_prior_scale):
                    # Sample latent variable from prior.
                    if self.z_prior_distribution == 'Normal':
                        z = pyro.sample("latent", dist.Normal(
                                torch.zeros(self.z_dim), torch.ones(self.z_dim)
                                ).to_event(1))
                    elif self.z_prior_distribution == 'Laplace':
                        z = pyro.sample("latent", dist.Laplace(
                            torch.zeros(self.z_dim), torch.ones(self.z_dim)
                            ).to_event(1))

                # Decode latent sequence.
                decoded = self.decoder(z, W, B, inverse_temp)
                if self.indel_factor_dependence:
                    insert_logits = decoded['insert_logits']
                    delete_logits = decoded['delete_logits']

                # Construct HMM parameters.
                if self.substitution_matrix:
                    initial_logits, transition_logits, observation_logits = (
                        self.statearrange(decoded['precursor_seq_logits'],
                                          decoded['insert_seq_logits'],
                                          insert_logits, delete_logits,
                                          substitute))
                else:
                    initial_logits, transition_logits, observation_logits = (
                        self.statearrange(decoded['precursor_seq_logits'],
                                          decoded['insert_seq_logits'],
                                          insert_logits, delete_logits))
                # Draw samples.
                if self.length_model:
                    pyro.sample("obs_L", dist.Poisson(decoded['L_mean']),
                                obs=L_data)
                pyro.sample("obs_seq",
                            MissingDataDiscreteHMM(initial_logits,
                                                   transition_logits,
                                                   observation_logits),
                            obs=seq_data)

    def guide(self, seq_data, L_data, local_scale, local_prior_scale):
        # Register encoder with pyro.
        pyro.module("encoder", self.encoder)

        # ARD weightings.
        if self.ARD_prior:
            alpha_conc = pyro.param("alpha_conc", torch.randn(self.z_dim))
            alpha_rate = pyro.param("alpha_rate", torch.randn(self.z_dim))
            pyro.sample("alpha", dist.Gamma(softplus(alpha_conc),
                                            softplus(alpha_rate)).to_event(1))
        # Factors.
        W_q_mn = pyro.param("W_q_mn", torch.randn([
                    self.z_dim, self.total_factor_size]))
        W_q_sd = pyro.param("W_q_sd", torch.ones([
                    self.z_dim, self.total_factor_size]))
        pyro.sample("W", dist.Normal(W_q_mn, softplus(W_q_sd)).to_event(2))
        B_q_mn = pyro.param("B_q_mn", torch.randn(self.total_factor_size))
        B_q_sd = pyro.param("B_q_sd", torch.ones(self.total_factor_size))
        pyro.sample("B", dist.Normal(B_q_mn, softplus(B_q_sd)).to_event(1))

        # Indel probabilities.
        if not self.indel_factor_dependence:
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

        # Substitution matrix.
        if self.substitution_matrix:
            substitute_q_mn = pyro.param("substitute_q_mn", torch.zeros(
                [self.latent_alphabet_length, self.alphabet_length]))
            substitute_q_sd = pyro.param("substitute_q_sd", torch.zeros(
                [self.latent_alphabet_length, self.alphabet_length]))
            pyro.sample("substitute", dist.Normal(
                    substitute_q_mn, softplus(substitute_q_sd)).to_event(2))

        # Per datapoint local latent variables.
        with pyro.plate("batch", L_data.shape[0]):
            # Encode sequences.
            z_loc, z_scale = self.encoder(seq_data)
            # Scale log likelihood to account for mini-batching.
            with poutine.scale(scale=local_scale*local_prior_scale):
                # Sample.
                if self.z_prior_distribution == 'Normal':
                    pyro.sample("latent",
                                dist.Normal(z_loc, z_scale).to_event(1))
                elif self.z_prior_distribution == 'Laplace':
                    pyro.sample("latent",
                                dist.Laplace(z_loc, z_scale).to_event(1))

    def fit_svi(self, dataset, epochs=2, anneal_length=1, batch_size=None,
                scheduler=None, jit=False):
        """Infer model parameters with stochastic variational inference."""

        # Setup.
        if batch_size is not None:
            self.batch_size = batch_size
        if scheduler is None:
            scheduler = MultiStepLR({'optimizer': Adam,
                                     'optim_args': {'lr': 0.01},
                                     'milestones': [],
                                     'gamma': 0.5})
        dataload = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Initialize guide.
        for seq_data, L_data in dataload:
            self.guide(seq_data, L_data, torch.tensor(1.), torch.tensor(1.))
            break
        # Setup stochastic variational inference.
        if jit:
            Elbo = JitTrace_ELBO(ignore_jit_warnings=True)
        else:
            Elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, scheduler, loss=Elbo)

        # Run inference.
        losses = []
        step_i = 1
        t0 = datetime.datetime.now()
        for epoch in range(epochs):
            for seq_data, L_data in dataload:
                loss = svi.step(
                    seq_data, L_data,
                    torch.tensor(dataset.data_size/L_data.shape[0]),
                    self._beta_anneal(step_i, batch_size, dataset.data_size,
                                      anneal_length))
                losses.append(loss)
                scheduler.step()
                step_i += 1
            print(epoch, loss, ' ', datetime.datetime.now() - t0)
        return losses

    def _beta_anneal(self, step, batch_size, data_size, anneal_length):
        """Annealing schedule for prior KL term (beta annealing)."""
        anneal_frac = step*batch_size/(anneal_length*data_size)
        return torch.tensor(min([anneal_frac, 1.]))

    def reconstruct_precursor_seq(self, data, ind, param):
        # Encode seq.
        z_loc = self.encoder(data[ind][0])[0]
        # Reconstruct
        decoded = self.decoder(z_loc, param("W_q_mn"), param("B_q_mn"),
                               param("inverse_temp_q_mn"))
        return torch.exp(decoded['precursor_seq_logits']).detach()
