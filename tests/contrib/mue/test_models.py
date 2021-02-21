# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from torch.optim import Adam

import pyro
from pyro.contrib.mue.dataloaders import BiosequenceDataset
from pyro.contrib.mue.models import FactorMuE, ProfileHMM
from pyro.optim import MultiStepLR


@pytest.mark.parametrize('length_model', [False, True])
@pytest.mark.parametrize('jit', [False, True])
def test_ProfileHMM_smoke(length_model, jit):
    # Setup dataset.
    seqs = ['BABBA', 'BAAB', 'BABBB']
    alph = ['A', 'B']
    dataset = BiosequenceDataset(seqs, 'list', alph)

    # Infer.
    scheduler = MultiStepLR({'optimizer': Adam,
                             'optim_args': {'lr': 0.1},
                             'milestones': [20, 100, 1000, 2000],
                             'gamma': 0.5})
    model = ProfileHMM(dataset.max_length, dataset.alphabet_length,
                       length_model)
    n_epochs = 5
    batch_size = 2
    losses = model.fit_svi(dataset, n_epochs, batch_size, scheduler, jit)

    assert not np.isnan(losses[-1])


@pytest.mark.parametrize('indel_factor_dependence', [False, True])
@pytest.mark.parametrize('z_prior_distribution', ['Normal', 'Laplace'])
@pytest.mark.parametrize('ARD_prior', [False, True])
@pytest.mark.parametrize('substitution_matrix', [False, True])
@pytest.mark.parametrize('length_model', [False, True])
@pytest.mark.parametrize('jit', [False, True])
def test_FactorMuE_smoke(indel_factor_dependence, z_prior_distribution,
                         ARD_prior, substitution_matrix, length_model, jit):
    # Setup dataset.
    seqs = ['BABBA', 'BAAB', 'BABBB']
    alph = ['A', 'B']
    dataset = BiosequenceDataset(seqs, 'list', alph)

    # Infer.
    z_dim = 2
    scheduler = MultiStepLR({'optimizer': Adam,
                             'optim_args': {'lr': 0.1},
                             'milestones': [20, 100, 1000, 2000],
                             'gamma': 0.5})
    model = FactorMuE(dataset.max_length, dataset.alphabet_length, z_dim,
                      indel_factor_dependence=indel_factor_dependence,
                      z_prior_distribution=z_prior_distribution,
                      ARD_prior=ARD_prior,
                      substitution_matrix=substitution_matrix,
                      length_model=length_model)
    n_epochs = 5
    anneal_length = 2
    batch_size = 2
    losses = model.fit_svi(dataset, n_epochs, anneal_length, batch_size,
                           scheduler, jit)

    # Reconstruct.
    recon = model.reconstruct_precursor_seq(dataset, 1, pyro.param)

    assert not np.isnan(losses[-1])
    assert recon.shape == (1, max([len(seq) for seq in seqs]), len(alph))

    assert torch.allclose(model._beta_anneal(3, 2, 6, 2), torch.tensor(0.5))
    assert torch.allclose(model._beta_anneal(100, 2, 6, 2), torch.tensor(1.))
